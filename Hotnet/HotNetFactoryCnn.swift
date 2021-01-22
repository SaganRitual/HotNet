// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

struct CNNLayerFactory {
    static func getActivation(
        _ standardized: HotNetConfiguration.Activation
    ) -> MPSCNNNeuron? {
        switch standardized {
        case .identity:
            return nil

        case .tanh:
            return MPSCNNNeuron(
                device: theDevice,
                neuronDescriptor: MPSNNNeuronDescriptor.cnnNeuronDescriptor(with: .tanH)
            )
        }
    }
}
