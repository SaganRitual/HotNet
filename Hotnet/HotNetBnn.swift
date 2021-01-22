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
        weights: UnsafeMutableRawPointer
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

        super.init(outputBuffer: outputBuffer)
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
        var layerIx = 0

        func activate_A() { HotNet.netDispatch.async(execute: activate_B) }

        func activate_B() {
            let layer = layers[layerIx]

            if layerIx == 0 { layer.activate(inputBuffer: input, activate_C) }
            else            { layer.activate(inputBuffer: intermediateBuffers[layerIx - 1], activate_C) }
        }

        func activate_C() {
            layerIx += 1

            if layerIx >= layers.count {
                DispatchQueue.main.async { onComplete(self.outputBuffer) }
                return
            }

            activate_A()
        }

        activate_A()
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
