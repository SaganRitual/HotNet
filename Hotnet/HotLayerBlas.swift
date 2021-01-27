import Accelerate

class HotLayerBlas {
    let cNeuronsIn: Int
    let cNeuronsOut: Int

    let activation: HotNetConfiguration.Activation

    let biases: UnsafePointer<Float>?
    let weights: UnsafePointer<Float>?
    let outputBuffer: UnsafeMutablePointer<Float>

    let typedOutputBuffer: UnsafeMutableBufferPointer<Float>

    init(
        cNeuronsIn: Int, cNeuronsOut: Int,
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer?,
        outputBuffer: UnsafeMutableRawPointer,
        activation: HotNetConfiguration.Activation = .tanh
    ) {
        self.cNeuronsIn = cNeuronsIn
        self.cNeuronsOut = cNeuronsOut
        self.activation = activation

        self.biases =
            HotLayerBlas.makeUnsafePointer(from: biases, cElements: cNeuronsOut)

        self.weights = HotLayerBlas.makeUnsafePointer(
            from: weights, cElements: cNeuronsIn * cNeuronsOut
        )

        self.outputBuffer = HotLayerBlas.makeUnsafeMutablePointer(
            from: outputBuffer, cElements: cNeuronsOut
        )!

        let t = outputBuffer.bindMemory(to: Float.self, capacity: cNeuronsOut)

        self.typedOutputBuffer =
            UnsafeMutableBufferPointer(start: t, count: cNeuronsOut)
    }

    func activate(inputBuffer: UnsafeRawPointer) {
        copyBiases()
        multiplyMatrices(inputBuffer)
        applyActivator()
    }
}

private extension HotLayerBlas {
    static func makeUnsafePointer(
        from pointer: UnsafeMutableRawPointer?, cElements: Int
    ) -> UnsafePointer<Float>? {
        guard let pointer = pointer else { return nil }
        let p = pointer.bindMemory(to: Float.self, capacity: cElements)
        return UnsafePointer(p)
    }

    static func makeUnsafeMutablePointer(
        from pointer: UnsafeMutableRawPointer?, cElements: Int
    ) -> UnsafeMutablePointer<Float>? {
        guard let pointer = pointer else { return nil }
        let p = pointer.bindMemory(to: Float.self, capacity: cElements)
        return UnsafeMutablePointer(p)
    }

    func applyActivator() {
        // I thought there would be a function in the blas library for applying
        // a function to each element. I guess not
        if activation == .identity { return }

        (0..<cNeuronsOut).forEach {
            typedOutputBuffer[$0] = tanh(typedOutputBuffer[$0])
        }
    }

    func copyBiases() {
        // The sgemv function does y = alpha * Ax + beta * y; copy
        // the biases to y here so the sgemv result will be added to them
        cblas_scopy(
            Int32(cNeuronsOut), // Number of elements in the vectors
            biases,             // Copy from biases vector
            Int32(1),           // Stride -- take each nth element, we want all
            outputBuffer,       // Copy to neurons output
            Int32(1)            // Stride for output -- write to each nth entry
        )
    }

    func multiplyMatrices(_ inputBuffer: UnsafeRawPointer) {
        let i = inputBuffer.bindMemory(to: Float.self, capacity: cNeuronsIn)
        let inputBuffer = UnsafePointer(i)

        cblas_sgemv(
            CblasRowMajor, CblasTrans,
            Int32(cNeuronsIn),  // Number of rows in A, that is, the weights
            Int32(cNeuronsOut), // Number of columns in A
            Float(1),           // alpha (scale for Ax result)
            weights,            // The matrix A
            Int32(cNeuronsOut), // Size of first dimension of A, aka "pitch", aka "lda"
            inputBuffer,        // The vector x
            Int32(1),           // Stride for x -- take each nth element, we want all
            Float(1),           // beta (scale for y)
            outputBuffer,       // The output vector y
            Int32(1)            // Stride for y -- write to each nth entry
        )
    }
}
