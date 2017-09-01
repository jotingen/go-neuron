package neuron

import (
	"math"
	"math/rand"
)

type Neuron struct {
	Weight   []float64 `json:"Weight"`
	Function string    `json:"Function"`
}

func (n *Neuron) Derivative(output float64) (derivative float64) {
	if n.Function == "RELU" {
		if output <= 0 {
			return 0.01
		} else {
			return 1.0
		}
	} else {
		return output * (1 - output)
	}
}

func (n *Neuron) Calc(input []float64) (output float64) {
	
	var net float64 = 0

	//Generate weights, add an extra for bias
	for len(n.Weight) <= len(input) {
		n.Weight = append(n.Weight, (float64(rand.Intn(256))-127)/127)
	}

	for i := range input {
		net += input[i] * n.Weight[i]
	}
	//Bias
	net += n.Weight[len(n.Weight)-1]

	if n.Function == "RELU" {
		if net <= 0 {
			return 0.01 * net
		} else {
			return net
		}
	} else {
		return 1 / (1 + math.Exp(-net))
	}
}
