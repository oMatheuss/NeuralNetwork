package com.sample;

import java.util.ArrayList;

public class NeuralNetwork {

	
	private ArrayList<float[][]> weights;
	private ArrayList<float[][]> biases;
	
	ArrayList<float[][]> layers;
	
	private float actualCost;
	
	public NeuralNetwork(int[] layer_sizes) {
		
		weights = new ArrayList<>();
		biases = new ArrayList<>();
		
		layers = new ArrayList<>();
		
		int weight_shapes[][] = new int[layer_sizes.length-1][2];
		for (int i = 1; i < layer_sizes.length; i++) {
			weight_shapes[i-1][0] = layer_sizes[i];
			weight_shapes[i-1][1] = layer_sizes[i-1];
		}
		
		for(int i = 0; i < weight_shapes.length; i++)
			weights.add(divideEach(rand(weight_shapes[i]), (float) Math.sqrt(weight_shapes[i][1])));
			//weights.add(rand(weight_shapes[i])); //for more than one layer, this works better
		
		for(int i = 1; i < layer_sizes.length; i++)
			biases.add(zeros(layer_sizes[i], 1));
		
	}
	
	//for testing a set of images accuracy
	public float accuracy(float[][] images, float[][] labels) {
		float cost = 0;
		float accuracy = 0;
		for (int i = 0; i < images.length; i++) {
			float[][] output = predict(vectorToMatrix(images[i]));
			float[][] label = vectorToMatrix(labels[i]);
			cost += cost(output, label);
			if(maxIndex(output) == maxIndex(label))
				accuracy++;
		}
		this.actualCost = cost/images.length;
		return accuracy/images.length*100;
	}
	
	//inputs -> layers -> output
	public float[][] predict(float[][] a) {
		for (int i = 0; i < weights.size(); i++) {
			
			a = add(dot(weights.get(i), a), biases.get(i));
			a = activation(a);
		}
		return a;
	}
	
	//look at cost formula
	public static float cost(float[][] output, float[][] target) {
		float[][] out_desired = subtract(output, target);
		float cost = 0;
		
		for (int i = 0; i < out_desired.length; i++)
			cost += (float) Math.pow(out_desired[i][0], 2);
		
		return cost;
	}
	
	//look at derivative cost formula
	public static float[][] d_cost(float[][] output, float[][] target) {
		float[][] out_desired = subtract(output, target);
		float[][] cost = new float[out_desired.length][out_desired[0].length];
		
		for (int i = 0; i < out_desired.length; i++)
			cost[i][0] =  2*out_desired[i][0];
		
		return cost;
	}
	
	//look at derivative cost formula for layer - 1
	//layer-1 -> weights(previous_weights) -> layer(d_act, d_cost) - to calculate d_cost of layer-1
	public static float[][] d_cost(float previous_weights[][], float[][] d_activation, float[][] d_cost) {
		float[][] a_t_c = directMult(d_activation, d_cost);
		
		float[][] new_d_cost = new float[previous_weights[0].length][1];
		
		for (int i = 0; i < previous_weights[0].length; i++) {
			for (int j = 0; j < previous_weights.length; j++)
				new_d_cost[i][0] += previous_weights[j][i]*a_t_c[j][0];
		}
		
		return new_d_cost;
	}
	
	//look at derivative ratio (cost/weight) formula for weight
	public static float[][] ratioInWeight(float previous_output[][], float[][] d_activation, float[][] d_cost) {
		float[][] a_t_c = directMult(d_activation, d_cost);
		
		float[][] ratio = dot(a_t_c, transpose(previous_output));
		
		return ratio;
	}
	
	
	//look at derivative ratio (cost/bias) formula for biases
	public static float[][] ratioInBias(float[][] d_activation, float[][] d_cost) {
		float[][] ratio = directMult(d_activation, d_cost);
		
		return ratio;
	}
	
	//activation function
	public static float[][] activation(float[][] a) {
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				a[i][j] = (float) (1/(1+Math.exp(-a[i][j])));
			}
		}
		return a;
	}
	
	//derivative of activation function
	//because im picking the values of each layer only after making the activation
	//just need to make the calculus below
	public static float[][] d_activation(float[][] a) {
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				a[i][j] = a[i][j]*(1-a[i][j]);
			}
		}
		return a;
	}
	
	//train the NN only after predicting all new w and b for received images and labels
	public void miniBatch(float[][] images, float[][] labels, int epochs) {
		
		ArrayList<float[][]> t_weights = new ArrayList<>();
		ArrayList<float[][]> t_biases = new ArrayList<>();
		
		for (float[][] f : weights) {
			float[][] a = new float[f.length][f[0].length];
			t_weights.add(a);
		}
		
		for (float[][] f : biases) {
			float[][] a = new float[f.length][f[0].length];
			t_biases.add(a);
		}
		
		ArrayList<float[][]> layers = new ArrayList<>();
		for (int i = 0; i < epochs; i++) {
			layers.clear();
			layers.add(vectorToMatrix(images[i]));
			
			float[][] temp = layers.get(0);
			for (int j = 0; j < weights.size(); j++) {
				
				temp = add(dot(weights.get(j), temp), biases.get(j));
				temp = activation(temp);
				layers.add(temp);
			}
			
			float[][] d_activation = d_activation(layers.get(layers.size()-1));
			float[][] d_cost = d_cost(layers.get(layers.size()-1), vectorToMatrix(labels[i]));
			
			float[][] ratioWeight = ratioInWeight(layers.get(layers.size()-2), d_activation, d_cost);
			float[][] ratioBias = ratioInBias(d_activation, d_cost);
			
			t_weights.set(
					t_weights.size()-1, 
					add(t_weights.get(t_weights.size()-1), ratioWeight));
			
			t_biases.set(t_biases.size()-1,
					add(t_biases.get(t_biases.size()-1), ratioBias));
			
			for (int j = 0; j < layers.size()-2; j++) {
				d_cost = d_cost(weights.get(weights.size()-1-j), d_activation, d_cost);
				d_activation = d_activation(layers.get(layers.size()-2-j));
				
				ratioWeight = ratioInWeight(layers.get(layers.size()-3-j), d_activation, d_cost);
				ratioBias = ratioInBias(d_activation, d_cost);
				
				t_weights.set(
						t_weights.size()-2-j,
						add(t_weights.get(t_weights.size()-2-j), ratioWeight));
				
				t_biases.set(t_biases.size()-2-j,
						add(t_biases.get(t_biases.size()-2-j), ratioBias));
			}
		}
		
		float unit = 0;
		
		for (int k = 0; k < t_weights.size(); k++) {
			float[][] f = t_weights.get(k);
			float[][] g = new float[f.length][f[0].length];
			for (int i = 0; i < f.length; i++) {
				for (int j = 0; j < f[0].length; j++) {
					g[i][j] = f[i][j]/epochs;
					unit += Math.pow(g[i][j], 2);
				}
			}
			t_weights.set(k, g);
		}
		
		for (int k = 0; k < t_biases.size(); k++) {
			float[][] f = t_biases.get(k);
			float[][] g = new float[f.length][f[0].length];
			for (int i = 0; i < f.length; i++) {
				for (int j = 0; j < f[0].length; j++) {
					g[i][j] = f[i][j]/epochs;
					unit += Math.pow(g[i][j], 2);
				}
			}
			t_biases.set(k, g);
		}
		
		unit = (float) Math.sqrt(unit);
		unit = -unit;
		
		for (int k = 0; k < t_weights.size(); k++) {
			weights.set(k, add(
					weights.get(k),
					divideEach(t_weights.get(k), unit)));
		}
		
		for (int k = 0; k < t_biases.size(); k++) {
			biases.set(k, add(
					biases.get(k),
					divideEach(t_biases.get(k), unit)));
		}
	}
	
	public void train(float[][] images, float[][] labels, int epochs, int batch_tam) {
		for (int i = 0; i < epochs; i ++) {
			float[][] miniBatch_img = new float[batch_tam][784];
			float[][] miniBatch_lbl = new float[batch_tam][784];
			
			for (int j = 0; j < batch_tam; j++) {
				int l = (int) (Math.random() * images.length);
				miniBatch_img[j] = images[l];
				miniBatch_lbl[j] = labels[l];
			}
			
			miniBatch(miniBatch_img, miniBatch_lbl, batch_tam);
		}
	}
	
	public static float[][] divideEach(float[][] a, float b) {
		for(int i = 0; i < a.length; i++) {
			for(int j = 0; j < a[0].length; j++) {
				a[i][j] = a[i][j]/b;
			}
		}
		return a;
	}
	
	
	public static int maxIndex(float[] a) {
		int max = 0;
		for (int i = 1; i < a.length; i++) {
			if (a[i] > a[max])
				max = i;
		}
		return max;
	}
	public static int maxIndex(float[][] a) {
		int max = 0;
		for (int i = 1; i < a.length; i++) {
			if (a[i][0] > a[max][0])
				max = i;
		}
		return max;
	}
	
	public static float[][] vectorToMatrix(float[] a) {
		float[][] r = new float [a.length][1];
		for (int i = 0; i < a.length; i++) {
			r[i][0] = a[i];
		}
		return r;
	}
	
	public static float[][] ones(int a, int b) {
		float[][] r = new float[a][b];
		for(int i = 0; i < r.length; i++) {
			for (int j = 0; j < r[0].length; j++) {
				r[i][j] = 1;
			}
		}
		return r;
	}
	
	public static float[][] zeros(int a, int b) {
		return new float[a][b];
	}
	
	public static float[][] rand(int[] a) {
		float[][] r = new float[a[0]][a[1]];
		java.util.Random rand = new java.util.Random();
		
		for(int i = 0; i < a[0]; i++) {
			for(int j = 0; j < a[1]; j++) {
				r[i][j] = (float) rand.nextGaussian();
			}
		}
		
		return r;
	}
	
	public static float[][] add(float[][] a, float[][] b)  {
		if (a.length != b.length || a[0].length != b[0].length) {
			System.out.println("numero de colunas ou linhas diferentes! add");
		}
		float[][] result = new float[a.length][a[0].length];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[0].length; j++) {
				result[i][j] = a[i][j]+b[i][j];
			}
		}
		return result;
	}
	
	public static float[][] add(float[][] a, float b)  {
		float[][] result = new float[a.length][a[0].length];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[0].length; j++) {
				result[i][j] = a[i][j]+b;
			}
		}
		return result;
	}
	
	public static float[][] subtract(float[][] a, float[][] b)  {
		if (a.length != b.length || a[0].length != b[0].length)
			System.out.println("numero de colunas ou linhas diferentes! subtract");
		
		float[][] result = new float[a.length][a[0].length];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[0].length; j++) {
				result[i][j] = a[i][j] - b[i][j];
			}
		}
		return result;
	}
	
	public static float[][] dot(float[][] a, float[][] b) {
		if(a[0].length != b.length) {
			System.out.println("O numero de linhas da matriz A deve ser igual ao de colunas na B!");
			return null;
		}
		
		float[][] result = new float[a.length][b[0].length];
		
		for(int i = 0; i < a.length; i++) {
			for(int j = 0; j < a[0].length; j++) {
				for(int k = 0; k < b[0].length; k++) {
					result[i][k] += a[i][j] * b[j][k];
				}
			}
		}
		return result;
	}
	
	public static float[][] directMult(float[][] a, float[][] b) {
		for(int i = 0; i < a.length; i++) {
			for(int j = 0; j < a[0].length; j++) {
				a[i][j] *= b[i][j];
			}
		}
		return a;
	}
	
	public static float[][] mult(float[][] a, float b) {
		for(int i = 0; i < a.length; i++) {
			for(int j = 0; j < a[0].length; j++) {
				a[i][j] *= b;
			}
		}
		return a;
	}
	
	public static float[][] transpose(float[][] a) {
		float[][] r = new float[a[0].length][a.length];
		
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				r[j][i] = a[i][j];
			}
		}
		
		return r;
	}
	
	public static void print(float[] a) {
		for(int i = 0; i < a.length; i++) {
			System.out.print(a[i] + " ");
			System.out.println();
		}
	}
	
	public static void print(float[][] a) {
		for(int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				System.out.print(a[i][j] + " ");
			}
			System.out.println();
		}
	}
	
	public static void print(int[][] a) {
		for(int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				System.out.print(a[i][j] + ", ");
			}
			System.out.println();
		}
	}
	
	public float getActualCost() {
		return actualCost;
	}

	public static void main(String[] args) {
		float[][] a = {
				{1, 2, 3},
				{4, 5, 6}
				};
		print(transpose(a));
	}
}
