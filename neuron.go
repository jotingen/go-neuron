package neuron

import (
	"fmt"
	"math"
	"math/rand"
	"os"
)

type Neuron struct {
	Weight []float64 `json:"Weight"`
}

func (n *Neuron) Derivative(output float64) (derivative float64) {
	return output * (1 - output)
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

	return 1 / (1 + math.Exp(-net))
}
