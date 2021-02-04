// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

extension HotNetConfig {
    class Buffers {
        enum LayerState { case top, hidden }
        private var state = LayerState.top

        private let config: HotNetConfig
        private var layerIndex = 0

        private var pBiases: UnsafeMutableRawPointer?
        private var pWeights: UnsafeMutableRawPointer?

        private lazy var topLayerWeights: UnsafeMutableRawPointer = {
            let cWeights =
                config.layerDescriptors[0].cNeurons *
                config.layerDescriptors[1].cNeurons

            let cBytes = cWeights * MemoryLayout<Float>.size

            let p = UnsafeMutableRawPointer.allocate(
                byteCount: cBytes, alignment: MemoryLayout<Float>.alignment
            )

            p.initializeMemory(as: Float.self, repeating: 1, count: cWeights)

            return p
        }()

        init(
            _ config: HotNetConfig,
            _ pBiases: UnsafeMutableRawPointer?,
            _ pWeights: UnsafeMutableRawPointer?
        ) {
            self.config = config
            self.pBiases = pBiases
            self.pWeights = pWeights
        }

        deinit { topLayerWeights.deallocate() }
    }
}

extension HotNetConfig.Buffers {
    var isInputLayer: Bool       { layerIndex == 0 }
    var isFirstHiddenLayer: Bool { layerIndex == 1 }
    var isHiddenLayer: Bool      { !isInputLayer && !isOutputLayer }
    var isLastHiddenLayer: Bool  { layerIndex == config.layerDescriptors.count - 2 }
    var isOutputLayer: Bool      { layerIndex == config.layerDescriptors.count - 1 }

    var prev: HotNetConfig.LayerDescriptor?   { isInputLayer ? nil : config.layerDescriptors[layerIndex - 1] }
    var curr: HotNetConfig.LayerDescriptor    { config.layerDescriptors[layerIndex] }
    var next: HotNetConfig.LayerDescriptor?   { isOutputLayer ? nil : config.layerDescriptors[layerIndex + 1] }

    func advanceLayer() {
        if !isHiddenLayer { return }

        let cNeuronsThis = config.layerDescriptors[layerIndex].cNeurons
        let cNeuronsNextDown = config.layerDescriptors[layerIndex + 1].cNeurons
        let fSize = MemoryLayout<Float>.size

        if pBiases != nil { pBiases! += cNeuronsThis * fSize }

        if pWeights != nil {
            pWeights! += cNeuronsThis * cNeuronsNextDown * fSize
        }
    }

    func biases() -> UnsafeMutableRawPointer? {
        isHiddenLayer ? pBiases : nil
    }

    func weights() -> UnsafeMutableRawPointer? {
        if isInputLayer       { return topLayerWeights }
        else if isHiddenLayer { return pWeights }
        else { fatalError() }
    }
}
