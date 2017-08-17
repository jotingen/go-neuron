package main

import (
	"fmt"
	"math/rand"
	"network"
)

var (
	layers []int
)

func main() {
	layers = []int{3, 2, 1}

	n := network.New(layers)

	for i := len(layers) - 1; i >= 0; i-- {
		for m := 0; m < layers[i]; m++ {
			fmt.Printf("wt %d %d %6.5f\n", i, m, n.Neurons[i][m].Weight)
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

		n.Train([]float64{x, y}, target)

		if t%5 == 0 {
			fmt.Printf("\r%6d", t)
			fmt.Printf("%10.8f[%4.2f]", n.Calc([]float64{0, 0}), 0.0-n.Calc([]float64{0, 0})[0])
			fmt.Printf("%10.8f[%4.2f]", n.Calc([]float64{0, 1}), 1.0-n.Calc([]float64{0, 1})[0])
			fmt.Printf("%10.8f[%4.2f]", n.Calc([]float64{1, 0}), 1.0-n.Calc([]float64{1, 0})[0])
			fmt.Printf("%10.8f[%4.2f]", n.Calc([]float64{1, 1}), 0.0-n.Calc([]float64{1, 1})[0])
		}
	}
	fmt.Println()

	for i := len(layers) - 1; i >= 0; i-- {
		for m := 0; m < layers[i]; m++ {
			fmt.Printf("wt %d %d %6.5f\n", i, m, n.Neurons[i][m].Weight)
		}
	}
}
