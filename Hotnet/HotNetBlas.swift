// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

class HotNetBlas: HotNet {
    let intermediateBuffers: [UnsafeMutableRawPointer]
    let layers: [HotLayerBlas]

    deinit {
        intermediateBuffers.forEach { $0.deallocate() }
    }

    init(
        _ configuration: HotNetConfiguration,
        biases: UnsafeMutableRawPointer,
        weights: UnsafeMutableRawPointer,
        callbackDispatch: DispatchQueue = DispatchQueue.main
    ) {
        (intermediateBuffers, layers) = HotNetBlas.makeLayers(
            configuration, biases: biases, weights: weights
        )

        // Point directly to the output buffer of the bottom layer, so
        // caller can read it without any copying
        let t = intermediateBuffers.last!.bindMemory(
            to: Float.self, capacity: configuration.layerDescriptors.last!.cNeurons
        )

        let outputBuffer = UnsafeBufferPointer(
            start: t, count: configuration.layerDescriptors.last!.cNeurons
        )

        super.init(outputBuffer: outputBuffer, callbackDispatch: callbackDispatch)
    }

    override func activate(
        input: UnsafeRawPointer
    ) -> UnsafeBufferPointer<Float> {
        activate_(input)
        return self.outputBuffer
    }

    override func activate(
        input: UnsafeRawPointer,
        _ onComplete: @escaping (UnsafeBufferPointer<Float>) -> Void
    ) {
        HotNet.netDispatch.async { [self] in
            activate_(input)
            callbackDispatch.async { onComplete(outputBuffer) }
        }
    }

    private func activate_(_ input: UnsafeRawPointer) {
        layers.first!.activate(inputBuffer: input)

        for (layer, buffer) in zip(layers.dropFirst(), intermediateBuffers) {
            layer.activate(inputBuffer: buffer)
        }
    }
}

private extension HotNetBlas {
    static func makeLayers(
        _ configuration: HotNetConfiguration,
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer?
    ) -> ([UnsafeMutableRawPointer], [HotLayerBlas]) {
        var intermediateBuffers = [UnsafeMutableRawPointer]()
        var pBiases = biases
        var pWeights = weights

        let layers: [HotLayerBlas] = zip(
            configuration.layerDescriptors.enumerated().dropLast(),
            configuration.layerDescriptors.dropFirst()
        ).map {
            let (upperLayerIndex, inputs) = $0
            let outputs = $1

            let cWeights = inputs.cNeurons * outputs.cNeurons
            let cBiases = outputs.cNeurons

            let intermediateBuffer = UnsafeMutableRawPointer.allocate(
                byteCount: outputs.cNeurons * MemoryLayout<Float>.size,
                alignment: MemoryLayout<Float>.alignment
            )

            intermediateBuffers.append(intermediateBuffer)

            defer {
                if !HotNet.isInputLayer(upperLayerIndex) {
                    pWeights = HotNet.advanceBufferPointer(
                        pElements: pWeights, cElements: cWeights
                    )
                }

                pBiases = HotNet.advanceBufferPointer(
                    pElements: pBiases, cElements: cBiases
                )
            }

            return HotLayerBlas(
                cNeuronsIn: inputs.cNeurons, cNeuronsOut: outputs.cNeurons,
                biases: pBiases, weights: pWeights,
                outputBuffer: intermediateBuffer,
                activation: configuration.activation
            )
        }

        return (intermediateBuffers, layers)
    }
}
