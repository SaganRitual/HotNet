// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class DataSourceCnn: NSObject, MPSCNNConvolutionDataSource {
    let pBiases: UnsafeMutablePointer<Float>
    let pWeights: UnsafeMutableRawPointer

    let convolutionDescriptor: MPSCNNConvolutionDescriptor

    init(
        weights: [Float], biases: [Float],
        convolutionDescriptor: MPSCNNConvolutionDescriptor
    ) {
        self.pWeights = UnsafeMutableRawPointer.allocate(
            byteCount: weights.count * MemoryLayout<Float>.size,
            alignment: MemoryLayout<Float>.alignment
        )

        self.pWeights.initializeMemory(
            as: Float.self, from: weights, count: weights.count
        )

        self.pBiases = UnsafeMutablePointer.allocate(capacity: biases.count)
        self.pBiases.initialize(from: biases, count: biases.count)

        self.convolutionDescriptor = convolutionDescriptor
    }

    func dataType() ->   MPSDataType { .float32 }
    func descriptor() -> MPSCNNConvolutionDescriptor { convolutionDescriptor }
    func weights() ->    UnsafeMutableRawPointer { pWeights }
    func biasTerms() ->  UnsafeMutablePointer<Float>? { pBiases }

    func load() -> Bool { true }

    func purge() { }

    func label() -> String? { nil }

    func copy(with zone: NSZone? = nil) -> Any { false }

}

