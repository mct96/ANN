#pragma once

#include <cassert>
#include <vector>
#include <array>
#include <memory>
#include <iostream>
#include <string>
#include <algorithm>
#include <random>
#include <iomanip>


#include "activation_function.hpp"

namespace ann {
using VectorD = std::vector<double>;

template <int T>
using ArrayD = std::array<double, T>;

class MLPNetwork {
public:
    MLPNetwork(
        std::vector<std::size_t> layersDesign,
        std::vector<ActivationFunction*> layersActivationFunction);

    ~MLPNetwork();
    void initializeTraining();

    VectorD predict(VectorD input);
    void train(VectorD input, VectorD d, double learningRate);

    void printWeights() const;
    void printInducedValues() const;
    void printSensibilities() const;
    void printLayersOutput() const;
    void print(std::string title, std::vector<VectorD> data) const;

private:
    double dotProduct(const VectorD& y, const VectorD& w, std::size_t startIdx);

    void initializeWeights();
    void initializeInducedValues();
    void initializeSensibilities();
    void initializeLayersOutput();

    VectorD forward(VectorD input);
    VectorD forwardB(VectorD input);

    VectorD calculateError(VectorD y, VectorD d) const;
    void calculateSensibility(VectorD err);
    void calculateOutputLayerSensibility(VectorD err);
    void calculateHiddenLayerSensibility();



    void updateWeights(double learningRate);

    std::size_t getNumNeurons(std::size_t layer) const;
    std::size_t getNumOutputNeurons() const;
    std::size_t getNumInputNeurons() const;

    VectorD forwardLayer(VectorD yj, std::size_t l);

    VectorD makeVector(std::size_t size, bool randon = false) const;

    // O todos os pesos da rede neural. Cada é um item do vetor. Todos os pesos
    // de uma camada estão contidas no mesmo array. Para cada neurônia há
    // (# de neurônios da camada anterior + 1) × (# neurônios da camada atual)
    // elementos no array.
    std::vector<VectorD> _weight;

    // Os valores de cada neurônio antes de ser calculado a função de ativação.
    std::vector<VectorD> _inducedValues;

    // O valor de sensibilidade de cada neurônio, exceto da camada de input.
    std::vector<VectorD> _sensibilityValues;

    std::vector<VectorD> _layersOutput;

    std::vector<std::size_t> _layerDesign;

    std::vector<ActivationFunction*> _activationFunction;

    std::size_t _nLayers;
    std::size_t _nHiddenLayers;

    // weight, induced values, sensibilities.
    std::size_t _nIntermediateLayers;
};

}