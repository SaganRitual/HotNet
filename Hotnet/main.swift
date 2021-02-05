// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation
import MetalPerformanceShaders

print("Hello, World!")

let semaphore = DispatchSemaphore(value: 0)
let theDevice = MTLCopyAllDevices()[0]

let callbackDispatch = DispatchQueue(
    label: "callback", attributes: .concurrent, target: DispatchQueue.global()
)

func fAlign() -> Int { MemoryLayout<Float>.alignment }
func fBytes(cElements: Int = 1) -> Int { MemoryLayout<Float>.size * cElements }

let configuration = HotNetConfig(
    activation: .identity,
    layerDescriptors: [
        .init(cNeurons: 9), .init(cNeurons: 9), .init(cNeurons: 9)
    ]
)

var cWeights: Int { configuration.counts.cWeights }
var cBiases: Int { configuration.counts.cBiases }

let pInputs = SwiftPointer(
    Float.self, elements: configuration.counts.perLayerCounts[0].cNeurons
)

let pInputs__: [Float] =
    (0..<configuration.counts.perLayerCounts[0].cNeurons).map { Float($0) }

let t = pInputs.getMutableBufferPointer().initialize(from: pInputs__)

let pWeights = UnsafeMutableRawPointer.allocate(
    byteCount: fBytes(cElements: cWeights), alignment: fAlign()
)

let f = (0..<cWeights).map { _ in Float.random(in: -1..<1)}
pWeights.initializeMemory(as: Float.self, from: f, count: cWeights)

let pBiases = UnsafeMutableRawPointer.allocate(
    byteCount: fBytes(cElements: cBiases), alignment: fAlign()
)

pBiases.initializeMemory(as: Float.self, repeating: 0, count: cBiases)

func blas() {

    let blas = HotNetBlas(
        configuration, biases: pBiases, weights: pWeights,
        callbackDispatch: callbackDispatch
    )

    blas.activate(input: pInputs.getRawPointer()) {
        print("blasResult \($0.map { $0 }) \($0.reduce(0) { $0 + $1 })")
        semaphore.signal()
    }
}

func bnn() {

    let bnn = HotNetBnn(
        configuration, biases: pBiases, weights: pWeights,
        callbackDispatch: callbackDispatch
    )

    bnn.activate(input: pInputs.getRawPointer()) {
        print("bnnResult \($0.map { $0 }) \($0.reduce(0) { $0 + $1 })")
        semaphore.signal()
    }
}

func cnn() {

    let cnn = HotNetCnn(
        configuration, biases: pBiases, weights: pWeights,
        callbackDispatch: callbackDispatch
    )

    cnn.activate(input: pInputs.getRawPointer()) {
        print("cnnResult \($0.map { $0 }) \($0.reduce(0) { $0 + $1 })")
        semaphore.signal()
    }
}

// We know what it feels like, thinking that it's a real possibility that
// we might never see our families again

blas()
semaphore.wait()

bnn()
semaphore.wait()

//cnn()
//semaphore.wait()
//
//let ann = AuxNet()
//let output32 = ann.activate()
//semaphore.wait()
//print("cnn2Result \(output32.map { $0 }) \(output32.reduce(0) { $0 + $1 })")
