#include "../include/mlp.hpp"

namespace ann {

MLPNetwork::MLPNetwork(
    std::vector<std::size_t> layersDesign,
    std::vector<ActivationFunction*> layersActivationFunction)
    :
    _layerDesign{layersDesign},
    _activationFunction{layersActivationFunction},
    _nLayers{layersDesign.size()},
    _nHiddenLayers{_nLayers - 2},
    _nIntermediateLayers{_nLayers - 1}
{

}

MLPNetwork::~MLPNetwork()
{

}

void MLPNetwork::initializeTraining()
{
    initializeWeights();
    initializeSensibilities();
    initializeInducedValues();
    initializeLayersOutput();
}

void MLPNetwork::initializeWeights()
{
    for (auto l = 1; l < _nLayers; ++l) {
        auto Sj = getNumNeurons(l-1);
        auto Si = getNumNeurons(l);

        auto Wi = (Sj + 1)*Si;
        auto weigths = makeVector(Wi, true);

        _weight.push_back(weigths);
    }
}

void MLPNetwork::initializeInducedValues()
{
    auto begin = ++_layerDesign.begin();
    auto end = _layerDesign.end();

    for (auto it = begin; it != end; ++it)
        _inducedValues.push_back(makeVector(*it, .5));
}

void MLPNetwork::initializeSensibilities()
{
    auto begin = ++_layerDesign.begin();
    auto end = _layerDesign.end();

    for (auto it = begin; it != end; ++it)
        _sensibilityValues.push_back(makeVector(*it));
}

void MLPNetwork::initializeLayersOutput()
{
    for (auto nNeurons: _layerDesign)
        _layersOutput.push_back(makeVector(nNeurons));
}

void MLPNetwork::train(VectorD inputLayer, VectorD d, double learningRate)
{
    VectorD y = forward(inputLayer);
    VectorD error = calculateError(y, d);
    calculateSensibility(error);
    updateWeights(learningRate);
}

void MLPNetwork::printWeights() const
{
    std::cout << "----------[[ WEIGHT ]]----------" << std::endl;

    for (int l = 0; l < _weight.size(); ++l) {
        auto Sj = _layerDesign[l];
        auto Si = _layerDesign[l + 1];

        std::cout << "LAYER " << l << ": " << std::endl;

        for (int i = 0; i < Si; ++i) {

            std::cout << "\tNEURON " << i << ": ";

            auto baseIdx = (Sj + 1) * i;
            for (int k = 0; k < (Sj + 1); ++k) {
                std::cout << _weight[l][baseIdx + k] << " ";
            }

            std::cout << std::endl;

        }

        std::cout << std::endl;
    }
}

double MLPNetwork::dotProduct(
    const VectorD& a, const VectorD& b, std::size_t startIdx)
{
    if (a.size() > (b.size() - startIdx))
        throw std::invalid_argument{"The vectors sizes are different."};

    double acc = 0;
    for (auto i = 0; i < a.size(); ++i)
        acc += a[i] * b[startIdx + i];

    return acc;
}

void MLPNetwork::printInducedValues() const
{
    print("INDUCED VALUES", _inducedValues);
}

void MLPNetwork::printSensibilities() const
{
    print("SENSIBILITIES", _sensibilityValues);
}

void MLPNetwork::printLayersOutput() const
{
    std::cout << "----------[[ OUTPUT ]]----------" << std::endl;

    int l = 0;
    for (auto layer: _layersOutput) {
        std::cout << "LAYER " << l++ << ":" << std::endl;
        std::cout << "\t";
        for (auto v: layer) std::cout << v << " ";
        std::cout << "\n" << std::endl;
    }

}

void MLPNetwork::print(std::string title, std::vector<VectorD> data) const
{
    std::cout << "----------[[ " << title << " ]]----------" << std::endl;

    for (int l = 0; l < data.size(); ++l) {
        auto Sj = _layerDesign[l];
        auto Si = _layerDesign[l + 1];

        std::cout << "LAYER " << l << ": " <<  std::endl;

        for (int i = 0; i < Si; ++i) {

            std::cout << "\tNEURON " << i << ": ";

            std::cout << data[l][i];

            std::cout << std::endl;

        }

        std::cout << std::endl;
    }
}

VectorD MLPNetwork::forward(VectorD inputLayer)
{
    auto yj = inputLayer;
    _layersOutput.at(0) = inputLayer;

    for (auto l = 1; l < _nLayers; ++l) {
        // Faz a propagação por uma camada.
        yj = forwardLayer(yj, l);

        // Armazena a saída dessa camada pra o treinamento.
        _layersOutput.at(l) = yj;
    }

    return yj;
}

VectorD MLPNetwork::predict(VectorD inputLayer)
{
    auto yj = inputLayer;

    for (auto l = 1; l < _nLayers; ++l) {
        auto Sj = getNumNeurons(l-1);
        auto Si = getNumNeurons(l);
        auto aFunc = _activationFunction.at(l-1);

        VectorD outputLayer = makeVector(Si);

        for (auto i = 0; i < Si; ++i) {
            auto baseIdx = i * (Sj + 1);

            double acc = _weight.at(l - 1).at(baseIdx); // bias.
            acc += dotProduct(yj, _weight.at(l - 1), baseIdx + 1);

            outputLayer.at(i) = aFunc->f(acc);
        }

        yj = outputLayer;
    }

    return yj;
}

VectorD MLPNetwork::forwardLayer(VectorD yj, std::size_t l)
{
    auto Sj = getNumNeurons(l-1);
    auto Si = getNumNeurons(l);

    VectorD yi{};

    for (auto i = 0; i < Si; ++i) {
        auto baseIdx = i * (Sj + 1);

        double acc = _weight.at(l-1).at(baseIdx); // bias.
        for (auto j = 0; j < Sj; ++j)
            acc += _weight.at(l-1).at(baseIdx + j + 1) * yj.at(j);

        _inducedValues.at(l-1).at(i) = acc;

        yi.push_back(_activationFunction.at(l-1)->f(acc));
    }

    return yi;
}

VectorD MLPNetwork::calculateError(VectorD y, VectorD d) const
{
    VectorD error{};

    for (auto i = 0; i < y.size(); ++i)
        error.push_back(d.at(i) - y.at(i));

    return error;
}

void MLPNetwork::calculateSensibility(VectorD errorVector)
{
    calculateOutputLayerSensibility(errorVector);
    calculateHiddenLayerSensibility();
}

void MLPNetwork::calculateOutputLayerSensibility(VectorD errorVector)
{
    auto Si = getNumOutputNeurons();

    auto activationFunction = _activationFunction.back();
    auto inducedValues = _inducedValues.back();

    VectorD outputSensibility = makeVector(Si);

    for (auto i = 0; i < Si; ++i) {
        auto secDev = activationFunction->Df(inducedValues.at(i));
        auto sensibility = errorVector.at(i) * secDev;

        outputSensibility.at(i) = sensibility;
    }

    _sensibilityValues.at(_nLayers - 2) = outputSensibility;
}

void MLPNetwork::calculateHiddenLayerSensibility()
{
    for (int l = _nLayers - 2; l > 0; --l)  {
        auto Sj = getNumNeurons(l - 1); // Previous layer.
        auto Si = getNumNeurons(l    ); // Current layer.
        auto Sk = getNumNeurons(l + 1); // Next layer.

        auto prevSensibilities = _sensibilityValues.at(l);
        auto weights = _weight.at(l);
        auto inducedValues = _inducedValues.at(l - 1);

        VectorD sensiblities = makeVector(Si);

        for (auto i = 0; i < Si; ++i) {
            auto acc = 0.0;

            for (auto k = 0; k < Sk; ++k) {
                auto baseIdx = k * (Si + 1);
                acc += (weights.at(baseIdx + i + 1)*prevSensibilities.at(k));
            }

            auto Df = _activationFunction.at(l-1)->Df(inducedValues.at(i));
            sensiblities.at(i) = acc * Df;
        }

        _sensibilityValues.at(l - 1) = sensiblities;
    }
}

void MLPNetwork::updateWeights(double learningRate)
{
    for (auto l = 1; l < _nLayers; ++l) {
        auto sensibilities = _sensibilityValues.at(l-1);
        auto layerOutput = _layersOutput.at(l-1);

        auto Sj = getNumNeurons(l - 1);
        auto Si = getNumNeurons(l    );

        for (auto i = 0; i < Si; ++i) {
            auto baseIdx = i * (Sj + 1);
            auto sensibility = sensibilities.at(i);

            _weight.at(l-1).at(baseIdx) += learningRate * sensibility; // bias.

            for (auto j = 0; j < Sj; ++j) {
                auto yi = layerOutput.at(j);
                auto dw = learningRate * sensibility * yi;
                _weight.at(l-1).at(baseIdx + j + 1) += dw;
            }
        }
    }
}

VectorD MLPNetwork::makeVector(std::size_t size, bool random) const
{
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0, 1);

    VectorD vector{};
    vector.assign(size, 0.0);

    if (random)
        for (auto& v: vector) v = dis(gen);

    return vector;
}

std::size_t MLPNetwork::getNumNeurons(std::size_t layer) const
{
    return _layerDesign.at(layer);
}

std::size_t MLPNetwork::getNumOutputNeurons() const
{
    return getNumNeurons(_nLayers-1);
}

std::size_t MLPNetwork::getNumInputNeurons() const
{
    return _layerDesign.at(0);
}


}