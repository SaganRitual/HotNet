//
//    Bloat16.swift
//    ZKit
//
//    The MIT License (MIT)
//
//    Copyright (c) 2016 Electricwoods LLC, Kaz Yoshikawa.
//
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in
//    all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
//    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//    THE SOFTWARE.
//

import Foundation
import Accelerate

struct Float16: CustomStringConvertible {

    var rawValue: UInt16

    static func float_to_float16(value: Float) -> UInt16 {
        let input: [Float] = [value]
        let output: [UInt16] = [0]

        let pInput = UnsafeMutableRawPointer.allocate(byteCount: MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment)
        pInput.initializeMemory(as: Float.self, from: input, count: input.count)

        let pOutput = UnsafeMutableRawPointer.allocate(byteCount: MemoryLayout<UInt16>.size, alignment: MemoryLayout<UInt16>.alignment)

        var sourceBuffer = vImage_Buffer(data: pInput, height: 1, width: 1, rowBytes: MemoryLayout<Float>.size)
        var destinationBuffer = vImage_Buffer(data: pOutput, height: 1, width: 1, rowBytes: MemoryLayout<UInt16>.size)
        vImageConvert_PlanarFtoPlanar16F(&sourceBuffer, &destinationBuffer, 0)

        defer { [pInput, pOutput].forEach { $0.deallocate() } }

        return output[0]
    }

    static func float16_to_float(value: UInt16) -> Float {
        let input: [UInt16] = [value]
        let output: [Float] = [0]

        let pInput = UnsafeMutableRawPointer.allocate(byteCount: MemoryLayout<UInt16>.size, alignment: MemoryLayout<UInt16>.alignment)
        pInput.initializeMemory(as: UInt16.self, from: input, count: input.count)

        let pOutput = UnsafeMutableRawPointer.allocate(byteCount: MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment)

        var sourceBuffer = vImage_Buffer(data: pInput, height: 1, width: 1, rowBytes: MemoryLayout<UInt16>.size)
        var destinationBuffer = vImage_Buffer(data: pOutput, height: 1, width: 1, rowBytes: MemoryLayout<Float>.size)
        vImageConvert_Planar16FtoPlanarF(&sourceBuffer, &destinationBuffer, 0)

        defer { [pInput, pOutput].forEach { $0.deallocate() } }

        return output[0]
    }

    static func floats_to_float16s(
        input: UnsafeRawPointer, output: UnsafeMutableBufferPointer<UInt16>
    ) {
        let width = vImagePixelCount(output.count)

        let pInput = UnsafeMutableRawPointer(mutating: input)
        var sourceBuffer = vImage_Buffer(
            data: pInput, height: 1, width: width,
            rowBytes: MemoryLayout<Float>.size * output.count
        )

        let pOutput = UnsafeMutableRawPointer(output.baseAddress)
        var destinationBuffer = vImage_Buffer(
            data: pOutput, height: 1, width: width,
            rowBytes: MemoryLayout<UInt16>.size * output.count
        )

        vImageConvert_PlanarFtoPlanar16F(&sourceBuffer, &destinationBuffer, 0)
    }

    static func floats_to_float16s(values: [Float]) -> [UInt16] {
        let inputs = values
        let width = vImagePixelCount(values.count)

        let pInput = UnsafeMutableRawPointer.allocate(byteCount: values.count * MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment)
        pInput.initializeMemory(as: Float.self, from: inputs, count: inputs.count)

        let pOutput = UnsafeMutableRawPointer.allocate(byteCount: values.count * MemoryLayout<UInt16>.size, alignment: MemoryLayout<UInt16>.alignment)

        var sourceBuffer = vImage_Buffer(data: pInput, height: 1, width: width, rowBytes: MemoryLayout<Float>.size * values.count)
        var destinationBuffer = vImage_Buffer(data: pOutput, height: 1, width: width, rowBytes: MemoryLayout<UInt16>.size * values.count)
        vImageConvert_PlanarFtoPlanar16F(&sourceBuffer, &destinationBuffer, 0)

        let pp = pOutput.assumingMemoryBound(to: UInt16.self)
        let ff = UnsafeBufferPointer(start: pp, count: values.count)

        defer { [pInput, pOutput].forEach { $0.deallocate() } }

        return ff.map { $0 }
    }

    static func float16s_to_floats(values: [UInt16]) -> [Float] {
        let inputs: [UInt16] = values
        let width = vImagePixelCount(values.count)

        let pInput = UnsafeMutableRawPointer.allocate(byteCount: MemoryLayout<UInt16>.size, alignment: MemoryLayout<UInt16>.alignment)
        pInput.initializeMemory(as: UInt16.self, from: inputs, count: inputs.count)

        let pOutput = UnsafeMutableRawPointer.allocate(byteCount: MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment)

        var sourceBuffer = vImage_Buffer(data: pInput, height: 1, width: width, rowBytes: MemoryLayout<UInt16>.size * values.count)
        var destinationBuffer = vImage_Buffer(data: pOutput, height: 1, width: width, rowBytes: MemoryLayout<Float>.size * values.count)
        vImageConvert_Planar16FtoPlanarF(&sourceBuffer, &destinationBuffer, 0)

        let pp = pOutput.assumingMemoryBound(to: Float.self)
        let ff = UnsafeBufferPointer(start: pp, count: values.count)

        defer { [pInput, pOutput].forEach { $0.deallocate() } }

        return ff.map { $0 }
    }

    init(_ value: Float) {
        self.rawValue = Float16.float_to_float16(value: value)
    }

    var floatValue: Float {
        return Float16.float16_to_float(value: self.rawValue)
    }

    var description: String {
        return self.floatValue.description
    }

    static func + (lhs: Float16, rhs: Float16) -> Float16 {
        return Float16(lhs.floatValue + rhs.floatValue)
    }

    static func - (lhs: Float16, rhs: Float16) -> Float16 {
        return Float16(lhs.floatValue - rhs.floatValue)
    }

    static func * (lhs: Float16, rhs: Float16) -> Float16 {
        return Float16(lhs.floatValue * rhs.floatValue)
    }

    static func / (lhs: Float16, rhs: Float16) -> Float16 {
        return Float16(lhs.floatValue / rhs.floatValue)
    }
}

