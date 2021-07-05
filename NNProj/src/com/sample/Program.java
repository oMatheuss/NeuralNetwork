package com.sample;

public class Program {
	
	public static void main(String[] args) {
		
		int[] layer_sizes = {784, 196, 10};
		
		//best for just one layer:
		//int[] layer_sizes = {784, 196, 10};
		NeuralNetwork net = new NeuralNetwork(layer_sizes);
		
		ReadMnist rm = new ReadMnist();

		float a = 0;
		int cont = 0;
		while (a < 90) {
			System.out.print("run " + cont + " - accuracy: ");
			net.train(rm.images, rm.labels, 10, 100);
			a = net.accuracy(rm.test_images, rm.test_labels);
			System.out.println(a + "% cost: " + net.getActualCost());
			cont++;
		}
	}
}
