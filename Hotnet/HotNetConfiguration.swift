// We are a way for the cosmos to know itself. -- C. Sagan

class HotNetConfiguration {
    let activation: Activation
    let counts: ParameterCounts
    let isAsync: Bool
    let layerDescriptors: [LayerDescriptor]
    let layerFactory: HotLayerFactoryProtocol

    init(
        activation: Activation, isAsync: Bool,
        layerDescriptors: [LayerDescriptor], layerFactory: HotLayerFactoryProtocol
    ) {
        self.activation = activation
        self.counts = HotNetConfiguration.computeNetParameters(layerDescriptors)
        self.isAsync = isAsync
        self.layerDescriptors = layerDescriptors
        self.layerFactory = layerFactory
    }

    enum Activation { case identity, tanh }

    struct LayerDescriptor {
        let activation: Activation
        let cNeurons: Int

        init(
            cNeurons: Int,
            _ activation: Activation = .tanh
        ) {
            self.activation = activation
            self.cNeurons = cNeurons
        }
    }

    struct ParameterCounts {
        let cBiases: Int
        let cNeurons: Int
        let cWeights: Int

        var cFixed: Int { cBiases + cWeights }

        init(_ cBiases: Int, _ cNeurons: Int, _ cWeights: Int) {
            self.cBiases = cBiases
            self.cNeurons = cNeurons
            self.cWeights = cWeights
        }
    }

    static func computeNetParameters(
        _ layerDescriptors: [LayerDescriptor]
    ) -> ParameterCounts {
        let dd = layerDescriptors

        let cNeurons = dd.reduce(0) { $0 + $1.cNeurons }
        let cBiases  = dd.dropFirst().reduce(0) { $0 + $1.cNeurons }

        let cWeights = zip(
            dd.dropLast(), dd.dropFirst()
        ).reduce(0) { $0 + ($1.0.cNeurons * $1.1.cNeurons) }

        return ParameterCounts(cBiases, cNeurons, cWeights)
    }

}
