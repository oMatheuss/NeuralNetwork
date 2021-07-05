package com.sample;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class ReadMnist {
	
	public float[][] images;
	public float[][] labels;
	public float[][] test_images;
	public float[][] test_labels;
	
	public ReadMnist() {
		
		File file = new File(getClass().getResource(
				"resources/train-images.idx3-ubyte").getPath());
		
		File file1 = new File(getClass().getResource(
				"resources/train-labels.idx1-ubyte").getPath());
		
		File file2 = new File(getClass().getResource(
				"resources/t10k-images.idx3-ubyte").getPath());
		
		File file3 = new File(getClass().getResource(
				"resources/t10k-labels.idx1-ubyte").getPath());
		
		images = imagesTofloatArray(file);
		labels = labelsTofloatArray(file1);
		test_images = imagesTofloatArray(file2);
		test_labels = labelsTofloatArray(file3);
	}
	
	
	
	public float[][] imagesTofloatArray(File file) {
		long size = file.length();
		System.out.println("Images size: " + size);
		
		byte[] conteudo = null;
		
		try {
			conteudo = Files.readAllBytes(file.toPath());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		float[][] r = new float[(int) ((size-16)/784)][784];
		
		long cont = 16;
		
		for (int k = 0; k < r.length; k ++) {
			for (int i = 0; i < 784; i++) {
				r[k][i] = Byte.toUnsignedInt(conteudo[(int) cont])/255.0f;
				cont++;
			}
		}
			
		return r;
	}
	
	public float[][] labelsTofloatArray(File file) {
		long size = file.length();
		System.out.println("Labels size: " + size);
		
		byte[] conteudo = null;
		
		try {
			conteudo = Files.readAllBytes(file.toPath());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		float[][] r = new float[(int) (size-8)][10];
		
		long cont = 8;
		
		for (int k = 0; k < r.length; k ++) {
			r[k][Byte.toUnsignedInt(conteudo[(int) cont])] = 1;
			cont++;
		}
			
		return r;
	}
}
