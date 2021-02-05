// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

extension HotLayerBnn {
    static func configureFullyConnectedLayer(
        cNeuronsIn: Int, cNeuronsOut: Int,
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer?,
        activation: BNNSActivation = .init(function: .tanh)
    ) -> BNNSFilter {
        let flags = BNNSNDArrayFlags(0)

        let cBiases = biases == nil ? 0 : cNeuronsOut
        let cWeightsIn = weights == nil ? 0 : cNeuronsIn
        let cWeightsOut = weights == nil ? 0 : cNeuronsOut

        let biasesDescription = BNNSNDArrayDescriptor(
            flags: flags, layout: BNNSDataLayoutVector,
            size: (cBiases, 0, 0, 0, 0, 0, 0, 0),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: biases, data_type: .float,
            table_data: nil, table_data_type: .float,
            data_scale: 0, data_bias: 0
        )

        let inDescription = BNNSNDArrayDescriptor(
            flags: flags, layout: BNNSDataLayoutVector,
            size: (cNeuronsIn, 0, 0, 0, 0, 0, 0, 0),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: nil, data_type: .float,
            table_data: nil, table_data_type: .float,
            data_scale: 0, data_bias: 0
        )

        let outDescription = BNNSNDArrayDescriptor(
            flags: flags, layout: BNNSDataLayoutVector,
            size: (cNeuronsOut, 0, 0, 0, 0, 0, 0, 0),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: nil, data_type: .float,
            table_data: nil, table_data_type: .float,
            data_scale: 0, data_bias: 0
        )

        let weightsDescription = BNNSNDArrayDescriptor(
            flags: flags, layout: BNNSDataLayoutRowMajorMatrix,
            size: (cWeightsIn, cWeightsOut, 0, 0, 0, 0, 0, 0),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: weights, data_type: .float,
            table_data: nil, table_data_type: .float,
            data_scale: 0, data_bias: 0)

        var layerParameters = BNNSLayerParametersFullyConnected(
            i_desc: inDescription, w_desc: weightsDescription,
            o_desc: outDescription, bias: biasesDescription,
            activation: activation
        )

        return BNNSFilterCreateLayerFullyConnected(&layerParameters, nil)!
    }

    static func configureInputPoolingLayer(
        kernelWidth: Int, kernelHeight: Int,
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer?,
        activation: BNNSActivation = .init(function: .tanh)
    ) -> BNNSFilter {
        let inDescriptor = BNNSNDArrayDescriptor(flags: BNNSNDArrayFlags(0),
                                                 layout: BNNSDataLayoutImageCHW,
                                                 size: (kernelWidth, kernelHeight, 1, 0, 0, 0, 0, 0),
                                                 stride: (0, 0, 0, 0, 0, 0, 0, 0),
                                                 data: nil,
                                                 data_type: .float,
                                                 table_data: nil,
                                                 table_data_type: .float,
                                                 data_scale: 0,
                                                 data_bias: 0)

        let outDescriptor = BNNSNDArrayDescriptor(flags: BNNSNDArrayFlags(0),
                                                  layout: BNNSDataLayoutImageCHW,
                                                  size: (kernelWidth, kernelHeight, 1, 0, 0, 0, 0, 0),
                                                  stride: (0, 0, 0, 0, 0, 0, 0, 0),
                                                  data: nil,
                                                  data_type: .float,
                                                  table_data: nil,
                                                  table_data_type: .float,
                                                  data_scale: 0,
                                                  data_bias: 0)

        var parameters = BNNSLayerParametersPooling(i_desc: inDescriptor,
                                                    o_desc: outDescriptor,
                                                    bias: BNNSNDArrayDescriptor(),
                                                    activation: .identity,
                                                    pooling_function: .average,
                                                    k_width: kernelWidth,
                                                    k_height: kernelHeight,
                                                    x_stride: 1,
                                                    y_stride: 1,
                                                    x_dilation_stride: 0,
                                                    y_dilation_stride: 0,
                                                    x_padding: 1,
                                                    y_padding: 1,
                                                    pad: (0, 0, 0, 0))

        return BNNSFilterCreateLayerPooling(&parameters, nil)!
    }
}
