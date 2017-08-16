package neuron

import (
	"fmt"
	"math"
	"math/rand"
	"os"
)

type Neuron struct {
	Weight []float64
	Output float64
	Error  float64
	Delta  float64
}

func (n *Neuron) Derivative() (derivative float64) {
	return n.Output * (1 - n.Output)
}

func (n *Neuron) Calc(input []float64) (output float64) {
	for _, v := range input {
		if v > 1 || v < -1 {
			fmt.Println("Error, input out of range:", v)
			os.Exit(1)
		}
	}

	var net float64 = 0

	//Generate weights, add an extra for bias
	for len(n.Weight) <= len(input) {
		n.Weight = append(n.Weight, ((rand.Float64() * 2) - 1))
	}

	for i := range input {
		net += input[i] * n.Weight[i]
	}
	//Bias
	net += n.Weight[len(n.Weight)-1]

	n.Output = 1 / (1 + math.Exp(-net))
	return n.Output
}
