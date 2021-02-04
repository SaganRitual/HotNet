// We are a way for the cosmos to know itself. -- C. Sagan

class HotNetConfig {
    let activation: Activation
    let counts: HotNetConfig.Counts
    let layerDescriptors: [LayerDescriptor]

    init(
        activation: Activation,
        layerDescriptors: [LayerDescriptor]
    ) {
        self.activation = activation
        self.layerDescriptors = layerDescriptors

        self.counts =
            HotNetConfig.Counts(layerDescriptors)
    }
}

extension HotNetConfig {
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
}
