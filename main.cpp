#include "include/mlp.hpp"
#include "include/activation_function.hpp"

#include <random>
#include <iomanip>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0, 1);

double sinNorm(double x) {
    return (sin(x) + 1) / 2;
}

double xorf(double a, double b) {
    return a != b ? 1 : 0;
}

int main() {
    auto logisticFunction = new ann::LogisticFunction{1};
    auto tanhFunction = new ann::HyperbolicTangentFunction{};
    auto arctanFunction = new ann::ArctanFunction{};

    ann::MLPNetwork network{
        {1, 32, 1},
        {arctanFunction, arctanFunction, arctanFunction, arctanFunction}};

    network.initializeTraining();


    for (int i = 0; i < 200000; ++i) {
        double x = dis(gen);
        double y = sin(x);

        network.train({x}, {y}, .1);
    }

    network.printWeights();
    network.printInducedValues();
    network.printSensibilities();
    network.printLayersOutput();

    std::cout << "-----------------------" << std::endl;

    for (int i = 0; i < 20; ++i) {
        auto x = dis(gen);
        auto y = sin(x);
        auto predicted = network.predict({x});
        auto error = predicted[0] - y;

        std::cout << std::setprecision(12);
        std::cout << "sin(" << std::setw(15) << x << ") = "
                  << std::setw(15) << y << " ANN says: "
                  << std::setw(15) << predicted[0] << " error of "
                  << std::setw(18) << error << std::endl;
    }

    return 0;
}