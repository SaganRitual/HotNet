// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

class HotNet {
    static let netDispatch = DispatchQueue(
        label: "net.dispatch.rob",
        attributes: .concurrent,
        target: DispatchQueue.global()
    )

    let callbackDispatch: DispatchQueue

    let outputBuffer: UnsafeBufferPointer<Float>

    init(
        outputBuffer: UnsafeBufferPointer<Float>,
        callbackDispatch: DispatchQueue
    ) {
        self.outputBuffer = outputBuffer
        self.callbackDispatch = callbackDispatch
    }
}
