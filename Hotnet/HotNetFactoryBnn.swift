// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

struct BNNLayerFactory {
    static func getActivation(
        _ standardized: HotNetConfiguration.Activation
    ) -> BNNSActivation {
        switch standardized {
        case .identity: return BNNSActivation(function: .identity)
        case .tanh:     return BNNSActivation(function: .tanh)
        }
    }
}
