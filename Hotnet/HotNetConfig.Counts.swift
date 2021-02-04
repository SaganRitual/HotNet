// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

extension HotNetConfig {
    struct Counts {
        let bufferAlignment = MemoryLayout<Float>.alignment

        let cBiases: Int
        let cNeurons: Int
        let cWeights: Int

        var cBiasesBytes: Int { cBiases * MemoryLayout<Float>.size }
        var cFixedBytes: Int { cFixed * MemoryLayout<Float>.size }
        var cNeuronsBytes: Int { cNeurons * MemoryLayout<Float>.size }
        var cWeightsBytes: Int { cWeights * MemoryLayout<Float>.size }

        var cFixed: Int { cBiases + cWeights }

        let perLayerCounts: [PerLayer]

        init(_ layerDescriptors: [LayerDescriptor]) {
            var counter = ComputeCounts(layerDescriptors: layerDescriptors)
            self.perLayerCounts = counter.compute()

            var cb = 0, cn = 0, cw = 0
            perLayerCounts.forEach {
                cb += $0.cBiases; cn += $0.cNeurons; cw += $0.cWeights
            }

            self.cBiases = cb; self.cNeurons = cn; self.cWeights = cw
        }
    }
}

extension HotNetConfig.Counts {
    struct PerLayer {
        let cNeurons: Int
        let cBiases: Int
        let cWeights: Int
    }
}

extension HotNetConfig {
    struct ComputeCounts {
        var isInputLayer: Bool       { layerIndex == 0 }
        var isFirstHiddenLayer: Bool { layerIndex == 1 }
        var isHiddenLayer: Bool      { !isInputLayer && !isOutputLayer }
        var isLastHiddenLayer: Bool  { layerIndex == layerDescriptors.count - 2 }
        var isOutputLayer: Bool      { layerIndex == layerDescriptors.count - 1 }

        var prev: LayerDescriptor?   { isInputLayer ? nil : layerDescriptors[layerIndex - 1] }
        var curr: LayerDescriptor    { layerDescriptors[layerIndex] }
        var next: LayerDescriptor?   { isOutputLayer ? nil : layerDescriptors[layerIndex + 1] }

        let layerDescriptors: [HotNetConfig.LayerDescriptor]

        var layerIndex = 0

        mutating func compute() -> [HotNetConfig.Counts.PerLayer] {
            (0..<layerDescriptors.count).map {
                self.layerIndex = $0

                return HotNetConfig.Counts.PerLayer(
                    cNeurons: curr.cNeurons,
                    cBiases: curr.cNeurons,
                    cWeights: curr.cNeurons * (prev?.cNeurons ?? 0)
                )
            }
        }
    }
}
