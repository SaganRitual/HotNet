// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

class HotNetBnn: HotNet {
    let layerFactory = BNNLayerFactory()

    let intermediateBuffers: [UnsafeMutableRawPointer]
    let layers: [HotLayerBnn]

    deinit {
        intermediateBuffers.forEach { $0.deallocate() }
    }

    init(
        _ configuration: HotNetConfiguration,
        biases: UnsafeMutableRawPointer,
        weights: UnsafeMutableRawPointer,
        callbackDispatch: DispatchQueue = DispatchQueue.main
    ) {
        (intermediateBuffers, layers) = HotNetBnn.makeLayers(
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

    func activate(input: [Float]) -> UnsafeBufferPointer<Float> {
        layers.indices.forEach {
            let layer = layers[$0]

            if $0 == 0 { layer.activate(inputBuffer: input) }
            else       { layer.activate(inputBuffer: intermediateBuffers[$0 - 1]) }
        }

        // Direct access to the motor outputs layer
        return self.outputBuffer
    }

    func activate(
        input: [Float],
        _ onComplete: @escaping (UnsafeBufferPointer<Float>) -> Void
    ) {
        HotNet.netDispatch.async { [self] in
            activate_(input)
            callbackDispatch.async { onComplete(outputBuffer) }
        }
    }

    func activate_(_ input: [Float]) {
        layers.first!.activate(inputBuffer: input)

        for (layer, buffer) in zip(layers.dropFirst(), intermediateBuffers) {
            layer.activate(inputBuffer: buffer)
        }
    }
}

private extension HotNetBnn {
    static func makeLayers(
        _ configuration: HotNetConfiguration,
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer?
    ) -> ([UnsafeMutableRawPointer], [HotLayerBnn]) {
        var intermediateBuffers = [UnsafeMutableRawPointer]()
        var pBiases = biases
        var pWeights = weights

        let layers: [HotLayerBnn] = zip(
            configuration.layerDescriptors.enumerated().dropLast(),
            configuration.layerDescriptors.enumerated().dropFirst()
        ).map {
            let (upperLayerIndex, inputs) = $0
            let (lowerLayerIndex, outputs) = $1

            let cWeights = inputs.cNeurons * outputs.cNeurons
            let cBiases = outputs.cNeurons

            let intermediateBuffer = UnsafeMutableRawPointer.allocate(
                byteCount: outputs.cNeurons * MemoryLayout<Float>.size,
                alignment: MemoryLayout<Float>.alignment
            )

            intermediateBuffers.append(intermediateBuffer)

            defer {
                if pWeights != nil && upperLayerIndex == 0 {
                    pWeights! += cWeights * MemoryLayout<Float>.size
                }

                if pBiases != nil &&
                   lowerLayerIndex < configuration.layerDescriptors.count - 1 {
                    pBiases! += cBiases * MemoryLayout<Float>.size
                }
            }

            return HotLayerBnn(
                cNeuronsIn: inputs.cNeurons, cNeuronsOut: outputs.cNeurons,
                biases: pBiases, weights: pWeights,
                outputBuffer: intermediateBuffer,
                activation: BNNLayerFactory.getActivation(configuration.activation)
            )
        }

        return (intermediateBuffers, layers)
    }
}
