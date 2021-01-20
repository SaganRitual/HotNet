// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

class HotNetBnn {
    let layers: [HotLayerBnn]
    var intermediateBuffers = [UnsafeMutableRawPointer]()

    var outputBuffer: UnsafeBufferPointer<Float>

    init(layout: [LayerOutputDescriptor], parameters: [Float]) {

        var intermediateBuffers = [UnsafeMutableRawPointer]()

        layers = zip(layout.dropLast(), layout.dropFirst()).map {
            (inputs, outputs) in

            let cWeights = inputs.cNeurons * outputs.cNeurons
            let cBiases = outputs.cNeurons

            let intermediateBuffer = UnsafeMutableRawPointer.allocate(
                byteCount: outputs.cNeurons * MemoryLayout<Float>.size,
                alignment: MemoryLayout<Float>.alignment
            )

            intermediateBuffers.append(intermediateBuffer)

            var parametersIt = parameters.makeIterator()

            return HotLayerBnn(
                cNeuronsIn: inputs.cNeurons, cNeuronsOut: outputs.cNeurons,
                weights: (0..<cWeights).map { _ in parametersIt.next()! },
                biases: (0..<cBiases).map  { _ in parametersIt.next()! },
                outputBuffer: intermediateBuffer,
                activationFunction: outputs.activationFunction
            )
        }

        self.intermediateBuffers = intermediateBuffers

        // Point directly to the output buffer of the bottom layer, so
        // caller can read it without any copying
        let t = intermediateBuffers.last!.bindMemory(
            to: Float.self, capacity: layout.last!.cNeurons
        )

        self.outputBuffer = UnsafeBufferPointer(
            start: t, count: layout.last!.cNeurons
        )
    }

    deinit {
        intermediateBuffers.forEach { $0.deallocate() }
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
