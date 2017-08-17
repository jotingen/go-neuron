package network

import (
	"fmt"
	"neuron"
)

type Network struct {
	Neurons [][]neuron.Neuron
}

var (
	neurons Network
)

func New(layer []int) *Network {

	n := new(Network)

	//Build neurons
	for l := range layer {
		var neuronLayer []neuron.Neuron
		for n := 0; n < layer[l]; n++ {
			neuronLayer = append(neuronLayer, neuron.Neuron{})
		}
		n.Neurons = append(n.Neurons, neuronLayer)
	}

	//Initialize weights
	n.Calc([]float64{0, 0})

	fmt.Println("YAY")
	return n
}

func (n *Network) Calc(inputs []float64) (outputs []float64) {
	for i := range n.Neurons {
		outputs = nil
		if i == 0 {
			//first layer uses inputs
			for m := range n.Neurons[i] {
				outputs = append(outputs, n.Neurons[i][m].Calc(inputs))
			}
		} else {
			//next layers use previous layer
			for m := range n.Neurons[i] {
				outputs = append(outputs, n.Neurons[i][m].Calc(inputs))
			}
		}
		inputs = outputs
	}
	return
}

func (n *Network) Train(inputs []float64, target []float64) {
	n.Calc(inputs)

	for i := len(n.Neurons) - 1; i >= 0; i-- {
		for m := 0; m < len(n.Neurons[i]); m++ {
			if i == len(n.Neurons)-1 {
				//Output Layer
				n.Neurons[i][m].Error = target[m] - n.Neurons[i][m].Output
				n.Neurons[i][m].Delta = n.Neurons[i][m].Error * n.Neurons[i][m].Derivative()

			} else {
				//Remaining Layers
				n.Neurons[i][m].Error = 0
				for j := 0; j < len(n.Neurons[i+1]); j++ {
					n.Neurons[i][m].Error += n.Neurons[i+1][j].Delta * n.Neurons[i+1][j].Weight[m]
				}
				n.Neurons[i][m].Delta = n.Neurons[i][m].Error * n.Neurons[i][m].Derivative()
			}
		}
	}

	learningRate := 1.0
	for i := len(n.Neurons) - 1; i >= 0; i-- {
		for m := 0; m < len(n.Neurons[i]); m++ {
			if i == 0 {
				//Input Layer
				for w := range inputs {
					n.Neurons[i][m].Weight[w] += learningRate * inputs[w] * n.Neurons[i][m].Delta
				}
			} else {
				//Remaining Layers
				for w := range n.Neurons[i][m].Weight {
					if w == len(n.Neurons[i][m].Weight)-1 {
						//Bias is last
						n.Neurons[i][m].Weight[w] += learningRate * 1 * n.Neurons[i][m].Delta
					} else {
						n.Neurons[i][m].Weight[w] += learningRate * n.Neurons[i-1][w].Output * n.Neurons[i][m].Delta
					}
				}
			}
		}
	}

}
