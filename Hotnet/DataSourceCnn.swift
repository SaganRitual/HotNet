// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class DataSourceCnn: NSObject, MPSCNNConvolutionDataSource {
    let pBiases: UnsafeMutablePointer<Float>?
    let pWeights: UnsafeMutableRawPointer?

    let convolutionDescriptor: MPSCNNConvolutionDescriptor

    init(
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer?,
        convolutionDescriptor: MPSCNNConvolutionDescriptor
    ) {
        self.pWeights = weights
        self.pBiases = biases == nil ? nil : biases!.assumingMemoryBound(to: Float.self)
        self.convolutionDescriptor = convolutionDescriptor
    }

    func dataType() ->   MPSDataType { .float32 }
    func descriptor() -> MPSCNNConvolutionDescriptor { convolutionDescriptor }
    func weights() ->    UnsafeMutableRawPointer { pWeights! }
    func biasTerms() ->  UnsafeMutablePointer<Float>? { pBiases }

    func load() -> Bool { true }

    func purge() { }

    func label() -> String? { nil }

    func copy(with zone: NSZone? = nil) -> Any { false }

}

