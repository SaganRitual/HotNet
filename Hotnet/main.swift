// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation
import MetalPerformanceShaders

print("Hello, World!")

let semaphore = DispatchSemaphore(value: 0)

struct BnnLayerOutputDescriptor {
    let activationFunction: BNNSActivation
    let cNeurons: Int

    init(
        _ activationFunction: BNNSActivationFunction = .tanh, _ cNeurons: Int
    ) {
        self.activationFunction = BNNSActivation(function: activationFunction)
        self.cNeurons = cNeurons
    }
}

func bnn() {
    let parameters = (0..<2).flatMap { _ in
        [Float](repeating: 1, count: 16) + [Float](repeating: 0, count: 4)
    }

    let bnn = HotNetBnn(
        layout: [
            BnnLayerOutputDescriptor(.identity, 4),
            BnnLayerOutputDescriptor(.identity, 4),
            BnnLayerOutputDescriptor(.identity, 4)
        ], parameters: parameters
    )

    let bnnResult = bnn.activate(input: [1, 2, 3, 4])
    print("bnnResult \(bnnResult.map { $0 })")

    let pOutput = UnsafeMutableRawPointer.allocate(
        byteCount: 4 * MemoryLayout<Float>.size,
        alignment: MemoryLayout<Float>.alignment
    )

    let layer = HotLayerBnn(
        cNeuronsIn: 4, cNeuronsOut: 4,
        weights: [Float](repeating: 1, count: 16),
        biases: [Float](repeating: 0, count: 4),
        outputBuffer: pOutput,
        activationFunction: .init(function: .identity)
    )

    layer.activate(inputBuffer: [Float]([1, 2, 3, 4]))

    let t = pOutput.assumingMemoryBound(to: Float.self)
    let f = UnsafeBufferPointer(start: t, count: 4)

    print("bnn layer result \(f.map { $0 })")
}

func cnn() {
    let gpus = MTLCopyAllDevices()
    let device = gpus[1]

    let parameters = (0..<2).flatMap { _ in
        [Float](repeating: 1, count: 16) + [Float](repeating: 0, count: 4)
    }

    let cnn = HotNetCnn(
        layout: [
            CnnLayerOutputDescriptor(nil, 4),
            CnnLayerOutputDescriptor(nil, 4),
            CnnLayerOutputDescriptor(nil, 4)
        ], parameters: parameters, isAsync: false
    )

    let cnnResult = cnn.activate(input: [1, 2, 3, 4])
    print("cnnResult \(cnnResult.map { $0 })")

    let pOutput = UnsafeMutableRawPointer.allocate(
        byteCount: 4 * MemoryLayout<Float>.size,
        alignment: MemoryLayout<Float>.alignment
    )

    let layer = HotLayerCnn(
        device: device, isAsync: false,
        cNeuronsIn: 4, cNeuronsOut: 4,
        weights: [Float](repeating: 1, count: 16),
        biases: [Float](repeating: 0, count: 4),
        outputBuffer: pOutput, activationFunction: nil
    )

    layer.activate(inputBuffer: [Float]([1, 2, 3, 4]))

    let t = pOutput.assumingMemoryBound(to: Float.self)
    let f = UnsafeBufferPointer(start: t, count: 4)

    print("cnn layer result \(f.map { $0 })")
    semaphore.signal()
}

bnn()
cnn()
semaphore.wait()
