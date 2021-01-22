// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

protocol HotLayerFactoryProtocol {
    func makeLayer(
        cNeuronsIn: Int, cNeuronsOut: Int,
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer,
        outputBuffer: UnsafeMutableRawPointer,
        activation: HotNetConfiguration.Activation
    ) -> HotLayerProtocol
}

protocol HotLayerProtocol {
    func activate(inputBuffer: UnsafeRawPointer)
}

class HotNet {
    let layers: [HotLayerProtocol]!
    let intermediateBuffers: [UnsafeMutableRawPointer]

    var outputBuffer: UnsafeBufferPointer<Float>
    var topLayerWeightsCuzBNNBarfsOnNil: UnsafeMutableBufferPointer<Float>

    let layerFactory = BNNLayerFactory()

    deinit {
        intermediateBuffers.forEach { $0.deallocate() }
        topLayerWeightsCuzBNNBarfsOnNil.deallocate()
    }

    init(
        _ configuration: HotNetConfiguration,
        biases: UnsafeMutableRawPointer,
        weights: UnsafeMutableRawPointer
    ) {
        topLayerWeightsCuzBNNBarfsOnNil = UnsafeMutableBufferPointer.allocate(
            capacity: configuration.layerDescriptors.first!.cNeurons * configuration.layerDescriptors.first!.cNeurons
        )

        vDSP.fill(&topLayerWeightsCuzBNNBarfsOnNil, with: 1)

        let topLayerWeights = UnsafeMutableRawPointer(
            topLayerWeightsCuzBNNBarfsOnNil.baseAddress
        )

        (self.intermediateBuffers, self.layers) =
            HotNet.makeLayers(
                configuration, biases: biases, weights: weights,
                topLayerWeights: topLayerWeights, layerFactory: layerFactory
            )

        // Point directly to the output buffer of the bottom layer, so
        // caller can read it without any copying
        let t = intermediateBuffers.last!.bindMemory(
            to: Float.self, capacity: configuration.layerDescriptors.last!.cNeurons
        )

        self.outputBuffer = UnsafeBufferPointer(
            start: t, count: configuration.layerDescriptors.last!.cNeurons
        )
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
}

private extension HotNet {
    static func makeLayers(
        _ configuration: HotNetConfiguration,
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer,
        topLayerWeights: UnsafeMutableRawPointer?,
        layerFactory: HotLayerFactoryProtocol
    ) -> ([UnsafeMutableRawPointer], [HotLayerProtocol]) {
        var intermediateBuffers = [UnsafeMutableRawPointer]()
        var pBiases = biases
        var pWeights = weights

        let layers: [HotLayerProtocol] = zip(
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
                if upperLayerIndex == 0 {
                    pWeights += cWeights * MemoryLayout<Float>.size
                }

                if pBiases != nil &&
                   lowerLayerIndex < configuration.layerDescriptors.count - 1 {
                    pBiases! += cBiases * MemoryLayout<Float>.size
                }
            }

            let pw: UnsafeMutableRawPointer =
                upperLayerIndex == 0 ? topLayerWeights! : pWeights

            return layerFactory.makeLayer(
                cNeuronsIn: inputs.cNeurons, cNeuronsOut: outputs.cNeurons,
                biases: pBiases, weights: pw,
                outputBuffer: intermediateBuffer,
                activation: configuration.activation
            )
        }

        return (intermediateBuffers, layers)
    }
}
