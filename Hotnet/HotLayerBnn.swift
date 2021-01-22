// We are a way for the cosmos to know itself. -- C. Sagan
// swiftlint:disable function_body_length
import Accelerate

class HotLayerBnn: HotLayerProtocol {
    let cNeuronsIn: Int
    let cNeuronsOut: Int

    let biasesDescription: BNNSNDArrayDescriptor
    let inDescription: BNNSNDArrayDescriptor
    let outDescription: BNNSNDArrayDescriptor
    let weightsDescription: BNNSNDArrayDescriptor

    let activation: BNNSActivation
    let filter: BNNSFilter
    var layerParameters: BNNSLayerParametersFullyConnected

    var outputBuffer: UnsafeMutableRawPointer

    init(
        cNeuronsIn: Int, cNeuronsOut: Int,
        biases: UnsafeMutableRawPointer?, weights: UnsafeMutableRawPointer,
        outputBuffer: UnsafeMutableRawPointer,
        activationFunction: BNNSActivation = .init(function: .tanh)
    ) {
        self.cNeuronsIn = cNeuronsIn
        self.cNeuronsOut = cNeuronsOut
        self.outputBuffer = outputBuffer
        self.activation = activationFunction

        let flags = BNNSNDArrayFlags(0)

        self.biasesDescription = BNNSNDArrayDescriptor(
            flags: flags, layout: BNNSDataLayoutVector,
            size: (cNeuronsOut, 0, 0, 0, 0, 0, 0, 0),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: biases, data_type: .float,
            table_data: nil, table_data_type: .float,
            data_scale: 0, data_bias: 0
        )

        self.inDescription = BNNSNDArrayDescriptor(
            flags: flags, layout: BNNSDataLayoutVector,
            size: (cNeuronsIn, 0, 0, 0, 0, 0, 0, 0),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: nil, data_type: .float,
            table_data: nil, table_data_type: .float,
            data_scale: 0, data_bias: 0
        )

        self.outDescription = BNNSNDArrayDescriptor(
            flags: flags, layout: BNNSDataLayoutVector,
            size: (cNeuronsOut, 0, 0, 0, 0, 0, 0, 0),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: nil, data_type: .float,
            table_data: nil, table_data_type: .float,
            data_scale: 0, data_bias: 0
        )

        self.weightsDescription = BNNSNDArrayDescriptor(
            flags: flags, layout: BNNSDataLayoutRowMajorMatrix,
            size: (cNeuronsIn, cNeuronsOut, 0, 0, 0, 0, 0, 0),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: weights, data_type: .float,
            table_data: nil, table_data_type: .float,
            data_scale: 0, data_bias: 0)

        self.layerParameters = BNNSLayerParametersFullyConnected(
            i_desc: inDescription, w_desc: weightsDescription,
            o_desc: outDescription, bias: biasesDescription,
            activation: activationFunction
        )

        self.filter = BNNSFilterCreateLayerFullyConnected(
            &layerParameters, nil
        )!
    }

    deinit { BNNSFilterDestroy(filter) }

    func activate(inputBuffer: UnsafeRawPointer) {
        BNNSFilterApply(filter, inputBuffer, outputBuffer)
    }
}
