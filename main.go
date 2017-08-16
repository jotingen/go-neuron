package main

import (
	"fmt"
	"math/rand"
	"network"
	"neuron"
)

var (
	neurons [][]neuron.Neuron
	layers  []int
)

func main() {
	layers = []int{3, 2, 1}

	network.New(layers)

	for l := range layers {
		var neuronLayer []neuron.Neuron
		for n := 0; n < layers[l]; n++ {
			neuronLayer = append(neuronLayer, neuron.Neuron{})
		}
		neurons = append(neurons, neuronLayer)
	}
	//Initialize weights
	calc([]float64{0, 0})

	for i := len(layers) - 1; i >= 0; i-- {
		for m := 0; m < layers[i]; m++ {
			fmt.Printf("wt %d %d %6.5f\n", i, m, neurons[i][m].Weight)
		}
	}

	for t := 0; t <= 1000000; t++ {

		x := rand.Float64()
		y := rand.Float64()
		var target []float64
		if (x >= .5 && y < .5) || (x < .5 && y >= .5) {
			target = append(target, 0.9)
		} else {
			target = append(target, 0.1)
		}

		train([]float64{x, y}, target)

		if t%5 == 0 {
			fmt.Printf("\r%6d", t)
			fmt.Printf("%10.8f[%4.2f]", calc([]float64{0, 0}), 0.0-calc([]float64{0, 0})[0])
			fmt.Printf("%10.8f[%4.2f]", calc([]float64{0, 1}), 1.0-calc([]float64{0, 1})[0])
			fmt.Printf("%10.8f[%4.2f]", calc([]float64{1, 0}), 1.0-calc([]float64{1, 0})[0])
			fmt.Printf("%10.8f[%4.2f]", calc([]float64{1, 1}), 0.0-calc([]float64{1, 1})[0])
		}
	}
	fmt.Println()

	for i := len(layers) - 1; i >= 0; i-- {
		for m := 0; m < layers[i]; m++ {
			fmt.Printf("wt %d %d %6.5f\n", i, m, neurons[i][m].Weight)
		}
	}
}

func calc(inputs []float64) (out []float64) {
	var layerOutputs [][]float64
	for i := range layers {
		var layerOutput []float64
		if i == 0 {
			//first layer uses inputs
			for m := 0; m < layers[i]; m++ {
				layerOutput = append(layerOutput, neurons[i][m].Calc(inputs))
			}
		} else {
			//next layers use previous layer
			for m := 0; m < layers[i]; m++ {
				layerOutput = append(layerOutput, neurons[i][m].Calc(layerOutputs[i-1]))
			}
		}
		layerOutputs = append(layerOutputs, layerOutput)
	}
	return layerOutputs[len(layers)-1]
}

func train(inputs []float64, target []float64) {
	calc(inputs)

	for i := len(layers) - 1; i >= 0; i-- {
		for m := 0; m < layers[i]; m++ {
			if i == len(layers)-1 {
				//Output Layer
				neurons[i][m].Error = target[m] - neurons[i][m].Output
				neurons[i][m].Delta = neurons[i][m].Error * neurons[i][m].Derivative()

			} else {
				//Remaining Layers
				neurons[i][m].Error = 0
				for j := 0; j < layers[i+1]; j++ {
					neurons[i][m].Error += neurons[i+1][j].Delta * neurons[i+1][j].Weight[m]
				}
				neurons[i][m].Delta = neurons[i][m].Error * neurons[i][m].Derivative()
			}
		}
	}

	learningRate := 1.0
	for i := len(layers) - 1; i >= 0; i-- {
		for m := 0; m < layers[i]; m++ {
			if i == 0 {
				//Input Layer
				for w := range inputs {
					neurons[i][m].Weight[w] += learningRate * inputs[w] * neurons[i][m].Delta
				}
			} else {
				//Remaining Layers
				for w := range neurons[i][m].Weight {
					if w == len(neurons[i][m].Weight)-1 {
						//Bias is last
						neurons[i][m].Weight[w] += learningRate * 1 * neurons[i][m].Delta
					} else {
						neurons[i][m].Weight[w] += learningRate * neurons[i-1][w].Output * neurons[i][m].Delta
					}
				}
			}
		}
	}

}
