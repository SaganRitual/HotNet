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

    func activate(input: UnsafeRawPointer) -> UnsafeBufferPointer<Float> {
        fatalError("Not in base class")
    }

    func activate(
        input: UnsafeRawPointer,
        _ onComplete: @escaping (UnsafeBufferPointer<Float>) -> Void
    ) { fatalError("Not in base class") }
}

extension HotNet {
    static func advanceBufferPointer(
        pElements: UnsafeMutableRawPointer?, cElements: Int
    ) -> UnsafeMutableRawPointer? {
        var advanced = pElements
        if advanced != nil {
            advanced! += cElements * MemoryLayout<Float>.size
        }

        return advanced
    }

    static func isInputLayer(_ index: Int) -> Bool { index == 0 }

    static func isOutputLayer(
        _ index: Int, _ configuration: HotNetConfiguration
    ) -> Bool {
        index >= configuration.layerDescriptors.count - 1
    }
}
