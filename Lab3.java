
/**
 * @Author: Yuting Liu and Jude Shavlik.  
 * 
 * Copyright 2017.  Free for educational and basic-research use.
 * 
 * The main class for Lab3 of cs638/838.
 * 
 * Reads in the image files and stores BufferedImage's for every example.  Converts to fixed-length
 * feature vectors (of doubles).  Can use RGB (plus grey-scale) or use grey scale.
 * 
 * You might want to debug and experiment with your Deep ANN code using a separate class, but when you turn in Lab3.java, insert that class here to simplify grading.
 * 
 * Some snippets from Jude's code left in here - feel free to use or discard.
 *
 */

import static java.lang.Math.random;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import javax.imageio.ImageIO;

public class Lab3 {

	public static int imageSize = 16; // Images are imageSize x imageSize. The
										// provided data is 128x128, but this
										// can be resized by setting this value
										// (or passing in an argument).
										// You might want to resize to 8x8,
										// 16x16, 32x32, or 64x64; this can
										// reduce your network size and speed up
										// debugging runs.
										// ALL IMAGES IN A TRAINING RUN SHOULD
										// BE THE *SAME* SIZE.

	public static enum Category {
		 butterfly,  flower, grand_piano, starfish,airplanes, watch ,
	}; // We'll hardwire these in, but more robust code would not do so.

	public static final Boolean useRGB = true; // If true, FOUR units are used
												// per pixel: red, green,
												// blue, and grey. If false,
												// only ONE (the grey-scale
												// value).
	public static int unitsPerPixel = (useRGB ? 4 : 1); // If using RGB, use
															// red+blue+green+grey.
															// Otherwise just
															// use the grey
															// value.

	private static String modelToUse = "deep"; // Should be one of {
													// "perceptrons",
													// "oneLayer", "deep" };
													// You
													// might want to use
													// this if
													// you are trying
													// approaches
													// other than a Deep
													// ANN.

	// Hongyi Wang maybe change it back to private when codes are copied in
	protected static int inputVectorSize; // The provided code uses a 1D vector
											// of
											// input features. You might want to
											// create a 2D version for your Deep
											// ANN
											// code.
											// Or use the get2DfeatureValue()
											// 'accessor function' that maps 2D
											// coordinates into the 1D vector.
											// The last element in this vector
											// holds
											// the 'teacher-provided' label of
											// the
											// example.
	// Hongyi Wang Maybe change it back to private when copied back
	protected static double eta = 0.01, fractionOfTrainingToUse = 1.00, dropoutRate = 0.5; // To
																							// turn
																							// off
																							// drop
																							// out,
																							// set
																							// dropoutRate
																							// to
																							// 0.0
																							// (or
																							// a
																							// neg
																							// number).
	// Hongyi Wang Maybe change it back to private when copied back
	protected static int maxEpochs = 200; // Feel free to set to a different
											// value.

	public static void main(String[] args) throws IOException {
		String trainDirectory = "images/trainset/";
		String tuneDirectory = "images/tuneset/";
		String testDirectory = "images/testset/";

		if (args.length > 5) {
			System.err.println(
					"Usage error: java Lab3 <train_set_folder_path> <tune_set_folder_path> <test_set_foler_path> <imageSize>");
			System.exit(1);
		}
		if (args.length >= 1) {
			trainDirectory = args[0];
		}
		if (args.length >= 2) {
			tuneDirectory = args[1];
		}
		if (args.length >= 3) {
			testDirectory = args[2];
		}
		if (args.length >= 4) {
			imageSize = Integer.parseInt(args[3]);
		}

		// Here are statements with the absolute path to open images folder
		File trainsetDir = new File(trainDirectory);
		File tunesetDir = new File(tuneDirectory);
		File testsetDir = new File(testDirectory);

		// create three datasets
		Dataset trainset = new Dataset();
		Dataset tuneset = new Dataset();
		Dataset testset = new Dataset();

		// Load in images into datasets.
		long start = System.currentTimeMillis();
		loadDataset(trainset, trainsetDir);
		System.out.println("The trainset contains " + comma(trainset.getSize()) + " examples.  Took "
				+ convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		start = System.currentTimeMillis();
		loadDataset(tuneset, tunesetDir);
		System.out.println("The  testset contains " + comma(tuneset.getSize()) + " examples.  Took "
				+ convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		start = System.currentTimeMillis();
		loadDataset(testset, testsetDir);
		System.out.println("The  tuneset contains " + comma(testset.getSize()) + " examples.  Took "
				+ convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		// Now train a Deep ANN. You might wish to first use your Lab 2 code
		// here and see how one layer of HUs does. Maybe even try your
		// perceptron code.
		// We are providing code that converts images to feature vectors. Feel
		// free to discard or modify.
		start = System.currentTimeMillis();
		trainANN(trainset, tuneset, testset);
		System.out
				.println("\nTook " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " to train.");

	}

	public static void loadDataset(Dataset dataset, File dir) {

		for (File file : dir.listFiles()) {
			// check all files
			if (!file.isFile() || !file.getName().endsWith(".jpg")) {
				continue;
			}
			// String path = file.getAbsolutePath();
			BufferedImage img = null, scaledBI = null;
			try {
				// load in all images
				img = ImageIO.read(file);
				// every image's name is in such format:
				// label_image_XXXX(4 digits) though this code could handle more
				// than 4 digits.
				String name = file.getName();
				int locationOfUnderscoreImage = name.indexOf("_image");

				// Resize the image if requested. Any resizing allowed, but
				// should really be one of 8x8, 16x16, 32x32, or 64x64 (original
				// data is 128x128).
				if (imageSize != 128) {
					scaledBI = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
					Graphics2D g = scaledBI.createGraphics();
					g.drawImage(img, 0, 0, imageSize, imageSize, null);
					g.dispose();
				}

				Instance instance = new Instance(scaledBI == null ? img : scaledBI,
						name.substring(0, locationOfUnderscoreImage));

				dataset.add(instance);
			} catch (IOException e) {
				System.err.println("Error: cannot load in the image file");
				System.exit(1);
			}
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////////////

	private static Category convertCategoryStringToEnum(String name) {
		if ("airplanes".equals(name))
			return Category.airplanes; // Should have been the singular
										// 'airplane' but we'll live with this
										// minor error.
		if ("butterfly".equals(name))
			return Category.butterfly;
		if ("flower".equals(name))
			return Category.flower;
		if ("grand_piano".equals(name))
			return Category.grand_piano;
		if ("starfish".equals(name))
			return Category.starfish;
		if ("watch".equals(name))
			return Category.watch;
		throw new Error("Unknown category: " + name);
	}

	// Hongyi Wang maybe change it back to private when codes are copied in
	protected static double getRandomWeight(int fanin, int fanout) {
	// This is one'rule of thumb'for initializing weights  for perceptrons and one-layerANN atleast.
		double range = Math.max(Double.MIN_VALUE, 4.0 / Math.sqrt(6.0 * (fanin + fanout)));
		return (2.0 * random() - 1.0) * range;
	}

	// Map from 2D coordinates (in pixels) to the 1D fixed-length feature
	// vector.
	private static double get2DfeatureValue(Vector<Double> ex, int x, int y, int offset) {
// If only using GREY, then offset = 0; Else offset = 0 for RED, 1 for GREEN, 2 for BLUE, and 3 for GREY.
		return ex.get(unitsPerPixel * (y * imageSize + x) + offset); // Jude: I
																		// have
																		// not
																		// used
																		// this,
																		// so
																		// might
																		// need
																		// debugging.
	}

	///////////////////////////////////////////////////////////////////////////////////////////////

	// Return the count of TESTSET errors for the chosen model.
	private static int trainANN(Dataset trainset, Dataset tuneset, Dataset testset) throws IOException {
		Instance sampleImage = trainset.getImages().get(0); // Assume there is
															// at least one
															// train image!

		inputVectorSize = sampleImage.getWidth() * sampleImage.getHeight() * unitsPerPixel + 1;
        // The '-1' for the bias is not explicitly added to all examples (instead code should implicitly handle it). The final 1 is for the CATEGORY

		// For RGB, we use FOUR input units per pixel: red, green, blue, plus
		// grey. Otherwise we only use GREY scale.
		// Pixel values are integers in [0,255], which we convert to a double in
		// [0.0, 1.0].
		// The last item in a feature vector is the CATEGORY, encoded as a
		// double in 0 to the size on the Category enum.
		// We do not explicitly store the '-1' that is used for the bias.
		// Instead code (to be written) will need to implicitly handle that
		// extra feature.
		System.out.println("\nThe input vector size is " + comma(inputVectorSize - 1) + ".\n");
		// 1D elements
		Vector<Vector<Double>> trainFeatureVectors = new Vector<Vector<Double>>(trainset.getSize());
		Vector<Vector<Double>> tuneFeatureVectors = new Vector<Vector<Double>>(tuneset.getSize());
		Vector<Vector<Double>> testFeatureVectors = new Vector<Vector<Double>>(testset.getSize());

		long start = System.currentTimeMillis();
		fillFeatureVectors(trainFeatureVectors, trainset);
		System.out.println("Converted " + trainFeatureVectors.size() + " TRAIN examples to feature vectors. Took "
				+ convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		start = System.currentTimeMillis();
		fillFeatureVectors(tuneFeatureVectors, tuneset);
		System.out.println("Converted " + tuneFeatureVectors.size() + " TUNE  examples to feature vectors. Took "
				+ convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		start = System.currentTimeMillis();
		fillFeatureVectors(testFeatureVectors, testset);
		System.out.println("Converted " + testFeatureVectors.size() + " TEST  examples to feature vectors. Took "
				+ convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		System.out.println("\nTime to start learning!");

		// Call your Deep ANN here. We recommend you create a separate class
		// file for that during testing and debugging, but before submitting
		// your code cut-and-paste that code here.

		if ("perceptrons".equals(modelToUse))
			return trainPerceptrons(trainFeatureVectors, tuneFeatureVectors, testFeatureVectors); // This
																									// is
																									// optional.
																									// Either
																									// comment
																									// out
																									// this
																									// line
																									// or
																									// just
																									// right
																									// a
																									// 'dummy'
																									// function.
		else if ("oneLayer".equals(modelToUse))
			return trainOneHU(trainFeatureVectors, tuneFeatureVectors, testFeatureVectors); // This
																							// is
																							// optional.
																							// Ditto.
		else if ("deep".equals(modelToUse))
			return trainDeep(trainFeatureVectors, tuneFeatureVectors, testFeatureVectors);
		return -1;
	}

	private static void fillFeatureVectors(Vector<Vector<Double>> featureVectors, Dataset dataset) {
		for (Instance image : dataset.getImages()) {
			featureVectors.addElement(convertToFeatureVector(image));
		}
	}

	private static Vector<Double> convertToFeatureVector(Instance image) {
		Vector<Double> result = new Vector<Double>(inputVectorSize);

		for (int index = 0; index < inputVectorSize - 1; index++) { // Need to
																	// subtract
																	// 1 since
																	// the last
																	// item is
																	// the
																	// CATEGORY.
			if (useRGB) {
				int xValue = (index / unitsPerPixel) % image.getWidth();
				int yValue = (index / unitsPerPixel) / image.getWidth();
				// System.out.println(" xValue = " + xValue + " and yValue = " +
				// yValue + " for index = " + index);
				if (index % 3 == 0)
					result.add(image.getRedChannel()[xValue][yValue] / 255.0); // If
																				// unitsPerPixel
																				// >
																				// 4,
																				// this
																				// if-then-elseif
																				// needs
																				// to
																				// be
																				// edited!
				else if (index % 3 == 1)
					result.add(image.getGreenChannel()[xValue][yValue] / 255.0);
				else if (index % 3 == 2)
					result.add(image.getBlueChannel()[xValue][yValue] / 255.0);
				else
					result.add(image.getGrayImage()[xValue][yValue] / 255.0); // Seems
																				// reasonable
																				// to
																				// also
																				// provide
																				// the
																				// GREY
																				// value.
			} else {
				int xValue = index % image.getWidth();
				int yValue = index / image.getWidth();
				result.add(image.getGrayImage()[xValue][yValue] / 255.0);
			}
		}
		result.add((double) convertCategoryStringToEnum(image.getLabel()).ordinal());
		// The last item is the CATEGORY, representing as an integer starting at 0 (and that int is then coerced to double).

		return result;
	}

	//////////////////// Some utility methods (cut-and-pasted from JWS'
	//////////////////// Utils.java file).
	//////////////////// ///////////////////////////////////////////////////

	private static final long millisecInMinute = 60000;
	private static final long millisecInHour = 60 * millisecInMinute;
	private static final long millisecInDay = 24 * millisecInHour;

	public static String convertMillisecondsToTimeSpan(long millisec) {
		return convertMillisecondsToTimeSpan(millisec, 0);
	}

	public static String convertMillisecondsToTimeSpan(long millisec, int digits) {
		if (millisec == 0) {
			return "0 seconds";
		} // Handle these cases this way rather than saying "0 milliseconds."
		if (millisec < 1000) {
			return comma(millisec) + " milliseconds";
		} // Or just comment out these two lines?
		if (millisec > millisecInDay) {
			return comma(millisec / millisecInDay) + " days and "
					+ convertMillisecondsToTimeSpan(millisec % millisecInDay, digits);
		}
		if (millisec > millisecInHour) {
			return comma(millisec / millisecInHour) + " hours and "
					+ convertMillisecondsToTimeSpan(millisec % millisecInHour, digits);
		}
		if (millisec > millisecInMinute) {
			return comma(millisec / millisecInMinute) + " minutes and "
					+ convertMillisecondsToTimeSpan(millisec % millisecInMinute, digits);
		}

		return truncate(millisec / 1000.0, digits) + " seconds";
	}

	public static String comma(int value) { // Always use separators (e.g.,
											// "100,000").
		return String.format("%,d", value);
	}

	public static String comma(long value) { // Always use separators (e.g.,
												// "100,000").
		return String.format("%,d", value);
	}

	public static String comma(double value) { // Always use separators (e.g.,
												// "100,000").
		return String.format("%,f", value);
	}

	public static String padLeft(String value, int width) {
		String spec = "%" + width + "s";
		return String.format(spec, value);
	}

	/**
	 * Format the given floating point number by truncating it to the specified
	 * number of decimal places.
	 * 
	 * @param d
	 *            A number.
	 * @param decimals
	 *            How many decimal places the number should have when displayed.
	 * @return A string containing the given number formatted to the specified
	 *         number of decimal places.
	 */
	public static String truncate(double d, int decimals) {
		double abs = Math.abs(d);
		if (abs > 1e13) {
			return String.format("%." + (decimals + 4) + "g", d);
		} else if (abs > 0 && abs < Math.pow(10, -decimals)) {
			return String.format("%." + decimals + "g", d);
		}
		return String.format("%,." + decimals + "f", d);
	}

	/**
	 * Randomly permute vector in place.
	 *
	 * @param <T>
	 *            Type of vector to permute.
	 * @param vector
	 *            Vector to permute in place.
	 */
	public static <T> void permute(Vector<T> vector) {
		if (vector != null) { // NOTE from JWS (2/2/12): not sure this is an
								// unbiased permute; I prefer (1) assigning
								// random number to each element, (2) sorting,
								// (3) removing random numbers.
			// But also see
			// "http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle" which
			// justifies this.
			/*
			 * To shuffle an array a of n elements (indices 0..n-1): for i from
			 * n - 1 downto 1 do j <- random integer with 0 <= j <= i exchange
			 * a[j] and a[i]
			 */

			for (int i = vector.size() - 1; i >= 1; i--) { // Note from JWS
															// (2/2/12): to
															// match the above I
															// reversed the FOR
															// loop that Trevor
															// wrote, though I
															// don't think it
															// matters.
				int j = random0toNminus1(i + 1);
				if (j != i) {
					T swap = vector.get(i);
					vector.set(i, vector.get(j));
					vector.set(j, swap);
				}
			}
		}
	}

	public static Random randomInstance = new Random(638 * 838); // Change the
																	// 638 * 838
																	// to get a
																	// different
																	// sequence
																	// of random
																	// numbers.

	/**
	 * @return The next random double.
	 */
	public static double random() {
		return randomInstance.nextDouble();
	}

	/**
	 * @param lower
	 *            The lower end of the interval.
	 * @param upper
	 *            The upper end of the interval. It is not possible for the
	 *            returned random number to equal this number.
	 * @return Returns a random integer in the given interval [lower, upper).
	 */
	public static int randomInInterval(int lower, int upper) {
		return lower + (int) Math.floor(random() * (upper - lower));
	}

	/**
	 * @param upper
	 *            The upper bound on the interval.
	 * @return A random number in the interval [0, upper).
	 * @see Utils#randomInInterval(int, int)
	 */
	public static int random0toNminus1(int upper) {
		return randomInInterval(0, upper);
	}

	// Perceptron output calculation
	// dealt with last element
	private static double calcOutPut(Vector<Double> vals, Vector<Double> perceptron) {
		double output = 0;
		// vals.size-1 because the last one for the input is label
		for (int i = 0; i < vals.size() - 1; i++) {
			output += vals.get(i) * perceptron.get(i);
		}
		// plus the bias node value
		output -= perceptron.lastElement();
		output = sigmoid(output);
		return output;

	}

	private static double getOutPut(Vector<Double> inputs, Vector<Double> perceptron) {
		double output = calcOutPut(inputs, perceptron);
		// If it is above the threshold the value output is 1 else 0
		if (output > 0.5)
			output = 1;

		else
			output = 0;
		return output;
	}

	private static double sigmoid(double x) {
		return (1 / (1 + Math.pow(Math.E, (-1 * x))));
	}

	private static double calAccuracy(Vector<Vector<Double>> testVectors, Vector<Vector<Double>> perceptrons) {

		// Get accuracy for test set
		// count for correct predictions
		int count = 0;
		double accuracy = 0.0;
		// Get outputs
		for (Vector<Double> test : testVectors) {
			for (int i = 0; i < perceptrons.size(); i++) {
				// Get the specific perceptron
				Vector<Double> perceptron = perceptrons.get(i);

				double output = getOutPut(test, perceptron);
				if (output == 1 && i == test.lastElement())
					count++;
			}
		}
		// calculate accuracy
		accuracy = count * 1.0 / testVectors.size();
		return accuracy;

	}

	private static int trainPerceptrons(Vector<Vector<Double>> trainFeatureVectors,
			Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors) {
		// Used to print out the content of each picture
		// for(Vector<Double> e : trainFeatureVectors)
		// {
		// System.out.println("~~~~~~~~~~~" + e);
		// }
		Vector<Vector<Double>> perceptrons = new Vector<Vector<Double>>(Category.values().length); // One
																									// perceptron
																									// per
																									// category.

		for (int i = 0; i < Category.values().length; i++) {













			Vector<Double> perceptron = new Vector<Double>(inputVectorSize);
			// Note:
			// inputVectorSize includes the OUTPUT CATEGORY as the LAST element. That element in the perceptron will be the BIAS.
			perceptrons.add(perceptron);
			for (int indexWgt = 0; indexWgt < inputVectorSize; indexWgt++)
				perceptron.add(getRandomWeight(inputVectorSize, 1)); // Initialize
																		// weights.
		}

		if (fractionOfTrainingToUse < 1.0) { // Randomize list, then get the
												// first N of them.
			int numberToKeep = (int) (fractionOfTrainingToUse * trainFeatureVectors.size());
			Vector<Vector<Double>> trainFeatureVectors_temp = new Vector<Vector<Double>>(numberToKeep);

			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute,
											// but that is OK.
			for (int i = 0; i < numberToKeep; i++) {
				trainFeatureVectors_temp.add(trainFeatureVectors.get(i));
			}
			trainFeatureVectors = trainFeatureVectors_temp;
		}

		int trainSetErrors = Integer.MAX_VALUE, tuneSetErrors = Integer.MAX_VALUE,
				best_tuneSetErrors = Integer.MAX_VALUE, testSetErrors = Integer.MAX_VALUE, best_epoch = -1,
				testSetErrorsAtBestTune = Integer.MAX_VALUE;
		long overallStart = System.currentTimeMillis(), start = overallStart;
		// Count for early stopping
		int count = 0;
		double oldAccuracy = 0;
		double accuracy = 0;

		for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) { // Might
																						// still
																						// want
																						// to
																						// train
																						// after
																						// trainset
																						// error
																						// =
																						// 0
																						// since
																						// we
																						// want
																						// to
																						// get
																						// all
																						// predictions
																						// on
																						// the
																						// 'right
																						// side
																						// of
																						// zero'
																						// (whereas
																						// errors
																						// defined
																						// wrt
																						// HIGHEST
																						// output).
			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute,
											// but that is OK.

			// CODE NEEDED HERE!

			// mark
			// Go through all training inputs

			for (Vector<Double> input : trainFeatureVectors) {
				// Get the label
				Double label = input.lastElement();

				for (int i = 0; i < perceptrons.size(); i++) {
					// Skip the last one because it is the label
					Vector<Double> perceptron = perceptrons.get(i);
					// Make the expected output of the correct perceptron to be
					// 1
					double expectedLabel = 0.0;
					if (i == label)
						expectedLabel = 1.0;

					// Get the output
					double output = calcOutPut(input, perceptron);

					// Calculate error
					double error = expectedLabel - output;

					// Update weight of bias node
					double changeOfWeights = perceptron.lastElement();
					changeOfWeights += eta * error * (output) * (1 - output) * -1;
					perceptron.set(perceptron.size() - 1, changeOfWeights);

					// Change weight of normal input node
					for (int j = 0; j < perceptron.size() - 1; j++) {
						changeOfWeights = perceptron.get(j);
						changeOfWeights += eta * error * (output) * (1 - output) * input.get(j);
						perceptron.set(j, changeOfWeights);
					}

				}

			}
			// early stopping
			oldAccuracy = accuracy;
			accuracy = calAccuracy(tuneFeatureVectors, perceptrons);
			if (oldAccuracy > accuracy) {
				count++;
			}
			else {
				best_epoch = epoch;
				best_tuneSetErrors = (int) ((1 - accuracy)* tuneFeatureVectors.size());
				testSetErrorsAtBestTune = (int) Math
						.round((1 - calAccuracy(testFeatureVectors, perceptrons)) * testFeatureVectors.size());
			}

			if (count > 3 && accuracy > 0.55)
			{
				System.out.println("current best_tuneSetErrors : " + best_tuneSetErrors );
				System.out.println("------------------------------------------------------");
				break;
			}


			System.out.print("   count: " + count + " ");

			System.out.println("Done with Epoch # " + comma(epoch) + ".  Took "
					+ convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " ("
					+ convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
			reportPerceptronConfig(); // Print out some info after epoch, so you
										// can see what experiment is running in
										// a given console.
			System.out.println("current best_tuneSetErrors : " + best_tuneSetErrors );
			System.out.println("------------------------------------------------------");
			start = System.currentTimeMillis();
		}
		System.out.println(
				"\n***** Best tuneset errors = " + comma(best_tuneSetErrors) + " of " + comma(tuneFeatureVectors.size())
						+ " (" + truncate((100.0 * best_tuneSetErrors) / tuneFeatureVectors.size(), 2)
						+ "%) at epoch = " + comma(best_epoch) + " (testset errors = " + comma(testSetErrorsAtBestTune)
						+ " of " + comma(testFeatureVectors.size()) + ", "
						+ truncate((100.0 * testSetErrorsAtBestTune) / testFeatureVectors.size(), 2) + "%).\n");
		return testSetErrorsAtBestTune;
	}

	private static void reportPerceptronConfig() {
		System.out.println("***** PERCEPTRON: UseRGB = " + useRGB + ", imageSize = " + imageSize + "x" + imageSize
				+ ", fraction of training examples used = " + truncate(fractionOfTrainingToUse, 2) + ", eta = "
				+ truncate(eta, 2) + ", dropout rate = " + truncate(dropoutRate, 2));
	}

	//////////////////////////////////////////////////////////////////////////////////////////////// ONE
	//////////////////////////////////////////////////////////////////////////////////////////////// HIDDEN
	//////////////////////////////////////////////////////////////////////////////////////////////// LAYER

	private static boolean debugOneLayer = false; // If set true, more things
													// checked and/or printed
													// (which does slow down the
													// code).
	// Hongyi Wang
	// Maybe change it back to private after the codes are copied in
	// Initial value 250
	protected static int numberOfHiddenUnits = 250;

	private static int trainOneHU(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors) {
		long overallStart   = System.currentTimeMillis(), start = overallStart;
		int  trainSetErrors = Integer.MAX_VALUE, tuneSetErrors = Integer.MAX_VALUE, best_tuneSetErrors = Integer.MAX_VALUE, testSetErrors = Integer.MAX_VALUE, best_epoch = -1, testSetErrorsAtBestTune = Integer.MAX_VALUE;

		for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) { // Might still want to train after trainset error = 0 since we want to get all predictions on the 'right side of zero' (whereas errors defined wrt HIGHEST output).
			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.

			// CODE NEEDED HERE!

			System.out.println("Done with Epoch # " + comma(epoch) + ".  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
			reportOneLayerConfig(); // Print out some info after epoch, so you can see what experiment is running in a given console.
			start = System.currentTimeMillis();
		}

		System.out.println("\n***** Best tuneset errors = " + comma(best_tuneSetErrors) + " of " + comma(tuneFeatureVectors.size()) + " (" + truncate((100.0 *      best_tuneSetErrors) / tuneFeatureVectors.size(), 2) + "%) at epoch = " + comma(best_epoch)
				+ " (testset errors = "    + comma(testSetErrorsAtBestTune) + " of " + comma(testFeatureVectors.size()) + ", " + truncate((100.0 * testSetErrorsAtBestTune) / testFeatureVectors.size(), 2) + "%).\n");
		return testSetErrorsAtBestTune;
	}

	private static void reportOneLayerConfig() {
		System.out.println("***** ONE-LAYER: UseRGB = " + useRGB + ", imageSize = " + imageSize + "x" + imageSize
				+ ", fraction of training examples used = " + truncate(fractionOfTrainingToUse, 2) + ", eta = "
				+ truncate(eta, 2) + ", dropout rate = " + truncate(dropoutRate, 2) + ", number HUs = "
				+ numberOfHiddenUnits
		// + ", activationFunctionForHUs = " + activationFunctionForHUs + ",
		// activationFunctionForOutputs = " + activationFunctionForOutputs
		// + ", # forward props = " + comma(forwardPropCounter)
		);
		// for (Category cat : Category.values()) { // Report the output unit
		// biases.
		// int catIndex = cat.ordinal();
		//
		// System.out.print(" bias(" + cat + ") = " +
		// truncate(weightsToOutputUnits[numberOfHiddenUnits][catIndex], 6));
		// } System.out.println();
	}

	// private static long forwardPropCounter = 0; // Count the number of
	// forward propagations performed.

	//////////////////////////////////////////////////////////////////////////////////////////////// DEEP
	//////////////////////////////////////////////////////////////////////////////////////////////// ANN
	//////////////////////////////////////////////////////////////////////////////////////////////// Code


	private static int trainDeep(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors,
			Vector<Vector<Double>> testFeatureVectors) throws IOException {
		// You need to implement this method!
		System.out.println("eta" + eta + " dropoutrate: " + dropoutRate);
        Vector<Vector<Double>> minitrain_v = new Vector<>();
        Vector<Double> test = new Vector<>();
        for(int i = 0 ; i < inputVectorSize; i ++){
            test.add(1.0);
        }

       CNN cnn = new CNN(minitrain_v,inputVectorSize - 1,5, 2, 20);
		double [] fractions = {0.25, 0.5, 0.75,1};

		for (int fraction_idx = 0; fraction_idx < fractions.length; fraction_idx++) {
			double prev_accuracy = 0;
			for (int epoch = 0; epoch < maxEpochs; epoch++) {
				 Collections.shuffle(trainFeatureVectors);
				 minitrain_v.clear();
				for (int i = 0; i < Math.ceil(trainFeatureVectors.size() * fractions[fraction_idx]); i++) {
					minitrain_v.add(trainFeatureVectors.get(i));
					//	System.out.println(trainFeatureVectors.get(i).lastElement());
				}

				// minitrain_v.add(test);
				Collections.shuffle(minitrain_v);
				cnn.input = minitrain_v;
				cnn.train();
				double accuracy = CalculateAccuracy(tuneFeatureVectors, cnn, false);
				if((prev_accuracy > accuracy) & (accuracy > 0.65)) {
					break;
				}
				System.out.println("tune accuracy: " + accuracy + "@ epoch: " + epoch);
				prev_accuracy = accuracy;
			}

			// test the accuracy and print the confusion matrix
			double accuracy = CalculateAccuracy(testFeatureVectors, cnn, true);
			String msg = "test accuracy @fraction value: " + fractions[fraction_idx] + " is " + accuracy + "\n";
			FileWriter outfile = new FileWriter(new File("lab3_data5.csv"), true);
			outfile.write(msg);
			outfile.close();
			System.out.println(msg);
			printOutConfusionMatrix();
		}

		return -1;
	}

    private static double CalculateAccuracy(Vector<Vector<Double>> FeatureVectors, CNN cnn, boolean TestSet) {
        int correct = 0;
        for (Vector<Double> image : FeatureVectors) {
            Double label = image.lastElement();
            Vector<Vector<Double>> dummy_input = new Vector<>();
            dummy_input.add(image);
            // forward
            cnn.inputlayer.getOut(dummy_input);
            int maxidx = Integer.MIN_VALUE ; double max = Double.MIN_VALUE;
            for (int i = 0 ;  i < cnn.outputlayer.outputs.size(); i++) {
                Vector<Double> output = cnn.outputlayer.outputs.get(i);
                if(output.get(0) > max){
                    max = output.get(0);
                    maxidx = i;
                }
			}
        //    System.out.println("label" + label + "maxidx: " + maxidx + " max: " + max );
            if(maxidx == label.intValue()){
                correct ++;
            }
            if(TestSet){
				confusionMatrix[maxidx][label.intValue()] += 1;
			}
        }
      //  System.out.println("total correct in tune: " + correct);

        return 1.0*correct/FeatureVectors.size();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
	private static int [][] confusionMatrix = new int [Category.values().length][Category.values().length];

	private static void printOutConfusionMatrix(){
		for (int[] i : confusionMatrix) {
			for (int j : i) {
				System.out.print(j + "|");
			}
			System.out.println();
		}

	}
}


import java.util.*;

		import static java.lang.Math.max;
		import static java.lang.Math.random;

/**
 * Created by Zirui Tao on 3/4/2017.
 */
public class CNN {
	public Vector<Vector<Double>> input;
	public layer inputlayer;
	public layer outputlayer;
	List<layer> layers;
	public static int inputSize;
	public static int covSize;
	public static int pooling_Size;
	public static int NumofPerceptronsPerLayer;
	public static perceptron dummy_input;
	public static perceptron dummy_bias;
	public static Map<key, weight> map;

	public CNN(Vector<Vector<Double>> featureVectors, int inputSize, int covSize, int pooling_Size, int numofPerceptronsPerLayer) {
		input = featureVectors;
		CNN.inputSize = inputSize;
		CNN.covSize = covSize;
		CNN.pooling_Size = pooling_Size;
		dummy_input = new perceptron();
		dummy_bias = new perceptron();
		layers = new ArrayList<>();
		NumofPerceptronsPerLayer = numofPerceptronsPerLayer;
		map = new HashMap<key,weight>();

		inputlayer = new inputLayer();
		Vector<Vector<Double>> dummy_input = new Vector<>();
		// dummy_input.add(input.get(0));
		// inputlayer.getOut(dummy_input);
		layer curLayer = inputlayer;
		while(curLayer != null){
			layers.add(curLayer);
			if(curLayer.next == null){
				outputlayer = curLayer;
			}
			curLayer = curLayer.next;
		}
		// System.out.println("back prop ------------------");
		// outputlayer.backProp();
	}
	public CNN() {
		map = new HashMap<key,weight>();
		dummy_input = new perceptron();
		dummy_bias = new perceptron();
		inputlayer = new inputLayer();
		//inputlayer.getOut(null);
		layer curLayer = inputlayer;
		while(curLayer != null){
			if(curLayer.next == null){
				outputlayer = curLayer;
			}
			curLayer = curLayer.next;
		}
		// System.out.println("back prop ------------------");
		// outputlayer.backProp();
	}
	public void train(){
		for (Vector<Double> inputImage : input) {
			Double label = inputImage.lastElement();
			Vector<Vector<Double>> dummy_input = new Vector<>();
			dummy_input.add(inputImage);
			// forward
			inputlayer.getOut(dummy_input);
//            System.out.println("output: ");
//            for (Vector<Double> output : outputlayer.outputs) {
//                for (Double d : output) {
//                    System.out.print(d.doubleValue() + " ");
//                }
//            }
			// System.out.println();

			Double [] errs = new Double[Lab3.Category.values().length];
			for (int i = 0; i < Lab3.Category.values().length; i++) {
				if(i != label.intValue()){
					errs[i] = 0 - outputlayer.outputs.get(i).get(0) ;
				}
				else{
					errs[i] = 1 - outputlayer.outputs.get(i).get(0);
				}
			}
			//       System.out.println("errs: ");
//            for (int i = 0; i < errs.length; i++) {
//                System.out.print(errs[i].doubleValue() + " ");
//            }
			//  System.out.println();
			// backprop
			// populate the delta value
//            System.out.println("dellllllllllllllllllllllllllllllta");
			for (int j = 0 ; j <  outputlayer.deltas.size(); j++) {
				Double cur_out = outputlayer.outputs.get(j).get(0);
				Double activate_slope =  outputlayer.activationSlope(cur_out);
				outputlayer.deltas.get(j)[0] += errs[j] * activate_slope;
//                    outputlayer.deltas.get(j)[0] = 1.0;
				//           System.out.print(outputlayer.deltas.get(j)[0] + " ");
			}
			//   System.out.println();
			fullyConnectedLayer out  = (fullyConnectedLayer) outputlayer;
//            System.out.println("before backprop, outlayer.wgt[0][0][0][0]: " + out.weightmap.get(out.pl.get(0))[0][0][0][0].weight);
			outputlayer.backProp();
//            System.out.println("after  backprop, outlayer.wgt[0][0][0][0]: " + out.weightmap.get(out.pl.get(0))[0][0][0][0].weight);

		}
	}
}

abstract class layer {
	layer prev;
	layer next;
	protected String activationFunctionForHUs;

	protected String activationFunctionForOutputs;
	int outputSize;
	List<perceptron> pl;
	Vector<Vector<Double>> input;
	Vector<Vector<Double>> weightSums;
	Vector<Vector<Double>> outputs;
	List<Double[]> deltas;
	int perceptronNum;
	public abstract void getOut(Vector<Vector<Double>> image);
	public abstract void backProp();
	public void printOutput(int unitsPerPixel) {
		for (int c = 0; c < unitsPerPixel; c++) {
			System.out.println("Color : " + c);
//            if (c == 0) {
			for (int i = 0; i < outputSize; i++) {
				for (int j = 0; j < outputSize; j++) {
					Double cur_Val = outputs.get(0).get((i * outputSize + j) + c * outputSize * outputSize);
					System.out.print(cur_Val + " ");
				}
				System.out.println();
				System.out.println("------------------");
			}
//            } else {
//                System.out.println("......");
//            }
		}
	}

	protected Double activateHU(double weightedSum) {
		if ("LeakyReLU".equalsIgnoreCase(this.activationFunctionForHUs))
			return (weightedSum <= 0 ? 0.01 * weightedSum : weightedSum);
		if ("ReLU".equalsIgnoreCase(this.activationFunctionForHUs)) return Math.max(0.0, weightedSum);
		if ("Sigmoid".equalsIgnoreCase(this.activationFunctionForHUs)) return 1.0 / (1.0 + Math.exp(-weightedSum));
		System.out.println("Unknown activation function for HUs: " + this.activationFunctionForHUs);
		return -1.0;
	}
	protected void activateLayer() {
		for (int i = 0; i < outputs.size(); i++) {
			Vector<Double> cur_out = outputs.get(i);
			for (int j = 0; j < cur_out.size(); j++) {
				Double val = cur_out.get(j);
				if (this.next == null){
					val = activateOutputs(val);
				}
				else{
					val = activateHU(val);
				}
				cur_out.setElementAt(val, j);
			}
			outputs.setElementAt(cur_out, i);
		}
	}
	protected Double activateOutputs(Double weightedSum) {
		if ("Sigmoid".equalsIgnoreCase(activationFunctionForOutputs)) return 1.0 / (1.0 + Math.exp(-weightedSum));
		if ("Linear".equalsIgnoreCase(activationFunctionForOutputs)) return weightedSum;
		System.out.println("Unknown activation function for output units: " + activationFunctionForOutputs);
		return -1.0;
	}
	protected Double activationSlope(Double activation) {
		if ("LeakyReLU".equalsIgnoreCase(activationFunctionForHUs)) return (activation >= 0 ? 1.0 : 0.01);
		if ("ReLU".equalsIgnoreCase(activationFunctionForHUs)) return (activation >= 0 ? 1.0 : 0.00);
		if ("Sigmoid".equalsIgnoreCase(activationFunctionForHUs) || "Sigmoid".equalsIgnoreCase(activationFunctionForOutputs))
			return activation * (1.0 - activation);
		if ("Linear".equalsIgnoreCase(activationFunctionForOutputs)) return 1.0;
		System.out.println("Unknown activation function for HUs: " + activationFunctionForHUs);
		return -1.0;
	}
	public void deepCopyweightSums () {
		for (Vector<Double> output : outputs) {
			Vector <Double> copies = new Vector<>();
			for (Double d : output) {
				copies.add(new Double(d.doubleValue()));
			}
			weightSums.add(copies);
		}
	}
	protected void printdeltas() {
		int unitsPerPixel = Lab3.unitsPerPixel;
		if((prev instanceof fullyConnectedLayer)){
			unitsPerPixel = 1;
		}
		System.out.println("DELTAS--------------------");
		Double[] delta = prev.deltas.get(0);
		for( int c = 0; c < unitsPerPixel; c++){
			System.out.println("color :" + c);
			for (int i = 0; i < prev.outputSize; i++) {
				for (int j = 0; j < prev.outputSize; j++){
					Double delta_element = delta[c * prev.outputSize* prev.outputSize + i * prev.outputSize + j];
					System.out.print( delta_element +  " ");
				}
				System.out.println();
			}
		}
	}
	protected void derivativeDeltas(){
		for (int i = 0; i < deltas.size(); i++) {
			Double [] delta = deltas.get(i);
			Vector <Double> output = outputs.get(i);
			for (int j = 0; j < delta.length; j++) {
				delta[j] *= activationSlope(output.get(j));
			}
			deltas.set(i,delta);
		}
	}
	protected void clearDeltas() {
		for (int i = 0 ; i < deltas.size(); i++) {
			Double [] delta = deltas.get(i);
			for (int j = 0; j < delta.length; j++) {
				delta[j] = 0.0;
			}
			deltas.set(i,delta);
		}
	}
}
class inputLayer extends layer{
	public Map < perceptron, weight[][][]> weightmap;
	public inputLayer() {
		deltas = new ArrayList<>();
		weightmap = new HashMap<>();
		outputSize = (Lab3.imageSize) - CNN.covSize + 1;
		pl = new ArrayList<>();
		for (int i = 0; i < CNN.NumofPerceptronsPerLayer; i++) {
			perceptron p = new perceptron();
			pl.add(p);
			weight[][][] wgt = new weight[CNN.covSize][CNN.covSize + 1][Lab3.unitsPerPixel];
			for (int j = 0 ; j < Lab3.unitsPerPixel; j ++){
				for (int k = 0; k < wgt.length; k++) {
					for (int l = 0; l < wgt[0].length; l++) {
						wgt[k][l][j] = new weight(Lab3.unitsPerPixel* CNN.covSize * CNN.covSize, CNN.NumofPerceptronsPerLayer);
						// bias at wgt[0][CNN.covSize][0] presumably!
					}
				}
			}
			weightmap.put(p,wgt);
			Double [] Delta = new Double[outputSize * outputSize * Lab3.unitsPerPixel];
			for (int d = 0; d < Delta.length; d++) {
				Delta[d] = 0.0;
			}
			deltas.add(Delta);
		}
		activationFunctionForHUs = "LeakyReLU";
		next = new poolingLayer(this);
	}

	@Override
	public void getOut(Vector<Vector<Double>> image_wrapper) {
//        System.out.println("inputLayer:");
		Vector<Double> image = image_wrapper.get(0);
		this.input = image_wrapper;
		outputs = new Vector<Vector<Double>>();
		for (perceptron p : pl) {
			Vector<Double> sampleOut = new Vector<>();
			weight[][][] wgt = weightmap.get(p);

			for(int k = 0; k < Lab3.unitsPerPixel; k++){
				for (int j = 0; j < outputSize; j++) {
					for (int i = 0; i < outputSize; i++) {
						Double window_out = 0.0;
						for(int offsety = 0; offsety < CNN.covSize; offsety++){
							for(int offsetx = 0; offsetx < CNN.covSize; offsetx++){
								// System.out.println( "image size:" + image.size() + ", k: " + k + ", j: " + j + ", i: " + i +", offsety: " + offsety + ", offsetx: " + offsetx);
								// System.out.println(k * Lab3.imageSize * Lab3.imageSize + (i + offsetx + Lab3.imageSize *(j + offsety)));
								window_out+= image.get(k + (i + offsetx + Lab3.imageSize*(j + offsety))* Lab3.unitsPerPixel)
										* wgt[offsetx][offsety][k].weight;
							}
						}
						// bias
						window_out += wgt[0][CNN.covSize][0].weight;
						sampleOut.add(window_out);
					}
				}
			}
			outputs.add(sampleOut);

		}

		//       System.out.println("input layer: ");
		weightSums = new Vector<>();
		deepCopyweightSums();
		//       System.out.println("input layer before activate");
		activateLayer();
//        printOutput(Lab3.unitsPerPixel);
		next.getOut(outputs);
	}

	@Override
	public void backProp() {
//        System.out.println("backProp input layer: ");
		derivativeDeltas();
		Vector<Double> image = this.input.get(0);
		for (int pidx = 0; pidx < pl.size(); pidx ++) {
			weight[][][] wgt = weightmap.get(pl.get(pidx));
			Double [] cur_delta = deltas.get(pidx);
			for(int c = 0; c < Lab3.unitsPerPixel; c++){
				for(int x = 0; x < Lab3.imageSize; x++){
					for(int y=  0; y < Lab3.imageSize; y++){
						for(int a = 0; a < CNN.covSize; a ++){
							for(int b = 0; b < CNN.covSize; b ++){
								if((x - a >= 0) && (x - a < outputSize) && (y - b >= 0) && (y - b < outputSize)){
									Double image_elements = image.get(c + (y * Lab3.imageSize + x)*Lab3.unitsPerPixel);
									weight weight = wgt[a][b][c];
									weight.weight += Lab3.eta * cur_delta[c +
											((y - b)*outputSize + (x - a))*Lab3.unitsPerPixel] * image_elements;
								}
							}
						}
					}

				}
				// update bias weight
				for(int a = 0; a < outputSize; a++){
					for(int b = 0; b < outputSize; b++){
						wgt[0][CNN.covSize][0].weight += Lab3.eta * cur_delta[c + Lab3.unitsPerPixel* (a * outputSize + b)];
					}
				}
			}
		}
		clearDeltas();
		outputs.clear();

	}
}
class convLayer extends layer{
	public Map < perceptron, weight[][][][]> weightmap;
	public convLayer(layer prev) {
		deltas = new ArrayList<>(); // one Double per delta
		weightmap = new HashMap<>();
		this.prev = prev;
		outputSize = (prev.outputSize - CNN.covSize + 1);
		pl = new ArrayList<>();
		for (int pidx = 0; pidx < CNN.NumofPerceptronsPerLayer; pidx++) {
			weight[][][][] wgt = new weight[CNN.NumofPerceptronsPerLayer][CNN.covSize][CNN.covSize + 1][Lab3.unitsPerPixel]; // including bias weight
			perceptron p = new perceptron();
			pl.add(p);
			// initialize delta matrix
			Double[] delta = new Double[outputSize * outputSize * Lab3.unitsPerPixel];
			for(int c = 0; c < Lab3.unitsPerPixel; c++){
				for (int height = 0; height < outputSize; height++) {
					for (int width = 0; width < outputSize; width++) {
						delta[c + (height * outputSize + width) * Lab3.unitsPerPixel] = new Double(0);
					}
				}
			}
			deltas.add(delta);
			for (int i = 0; i < CNN.NumofPerceptronsPerLayer; i++) {
				for (int ii = 0; ii < CNN.covSize; ii++) {
					for (int j = 0; j < CNN.covSize + 1; j++) {
						for (int k = 0; k < Lab3.unitsPerPixel; k++) {
							wgt[i][ii][j][k] = new weight(prev.pl.size(), pl.size());
							// bias at wgt[0][0][CNN.covSize][0] presumably!
						}
					}
				}
				weightmap.put(p, wgt);
				// CNN.map.put(new key(CNN.dummy_bias,p), new weight(CNN.covSize* CNN.covSize, CNN.NumofPerceptronsPerLayer));
			}

		}
		activationFunctionForHUs = "LeakyReLU";
		next = new poolingLayer(this);
	}
	@Override
	public void getOut(Vector<Vector<Double>> input) {
		this.input = input;
//       System.out.println("Conv Layer:");
		outputs = new Vector<>();
		for (perceptron p : pl){
			Vector<Double> sampleOut = new Vector<>();
			weight[][][][] wgt = weightmap.get(p);
			for(int c = 0; c < Lab3.unitsPerPixel; c++){
				for(int y = 0; y < outputSize; y ++){
					for (int x = 0; x < outputSize; x++){
						Double window_out = 0.0;
						for (int b = 0; b < CNN.covSize; b ++){
							for (int a = 0; a < CNN.covSize; a ++){
								for(int i = 0; i< CNN.NumofPerceptronsPerLayer ; i++){
									Vector<Double> cur_plate = input.get(i);
									window_out+= cur_plate.get((x + a +(b + y)* prev.outputSize)*Lab3.unitsPerPixel + c)* wgt[i][b][a][c].weight;
								}
							}
							// bias at wgt[0][0][CNN.covSize][0] presumably!
						}
						window_out += wgt[0][0][CNN.covSize][0].weight;
						sampleOut.add(window_out);
					}
				}
			}
			outputs.add(sampleOut);
		}
		weightSums = new Vector<>();
		deepCopyweightSums();
		activateLayer();
//        printOutput(Lab3.unitsPerPixel);
		next.getOut(outputs);
	}

	@Override
	public void
	backProp() {
		derivativeDeltas();
		if(prev != null ) {
//            System.out.println("backProp conv layer");
			for (perceptron p : pl) {
				weight[][][][] wgt = weightmap.get(p);
				// bias at wgt[0][0][CNN.covSize][0] presumably!
				int idx = pl.indexOf(p);
				Double [] cur_delta = deltas.get(idx);
				for (int deltaidx = 0; deltaidx < prev.deltas.size(); deltaidx++){
					Double [] delta = prev.deltas.get(deltaidx);
					Vector<Double> output = prev.outputs.get(deltaidx);
					for(int c = 0; c < Lab3.unitsPerPixel; c++){
						for(int x = 0; x < prev.outputSize; x ++){
							for(int y = 0; y < prev.outputSize; y ++){
								for(int b = 0; b < CNN.covSize; b ++){
									for(int a = 0; a < CNN.covSize; a ++){
										if((x - a >= 0) && (x - a < outputSize) && (y - b >= 0) && (y - b < outputSize)){
											weight weight = wgt[deltaidx][b][a][c];
											Double cur_delta_element = cur_delta[c + Lab3.unitsPerPixel * ((y - b) * outputSize + (x - a))];
											delta[c + Lab3.unitsPerPixel * (y * prev.outputSize + x)] += cur_delta_element * weight.weight;
										}
									}
								}
							}
						}

					}
					// update weights
					for(int c = 0; c < Lab3.unitsPerPixel; c++) {
						for (int x = 0; x < prev.outputSize; x++) {
							for (int y = 0; y < prev.outputSize; y++) {
								for (int a = 0; a < CNN.covSize; a++) {
									for (int b = 0; b < CNN.covSize; b++) {
										if ((x - a >= 0) && (x - a < outputSize) && (y - b >= 0) && (y - b < outputSize)) {
											weight weight = wgt[deltaidx][a][b][c];
											Double output_elements = output.get(c + Lab3.unitsPerPixel * (x * prev.outputSize + y));
											weight.weight += Lab3.eta * cur_delta[c +
													Lab3.unitsPerPixel * ((y - b) * outputSize + (x - a))] * output_elements;
										}
									}
								}
							}
						}
					}
					prev.deltas.set(deltaidx,delta);
				}
				// bias at wgt[0][0][CNN.covSize][0] presumably!
				for (int i = 0; i < cur_delta.length; i++) {
					wgt[0][0][CNN.covSize][0].weight += Lab3.eta * cur_delta[i];
				}
			}
//            printdeltas();
			prev.backProp();
		}
		clearDeltas();
		outputs.clear();
	}
}

class poolingLayer extends layer{
	static int count = 0 ;
	//List<List<List<weight>>> weightMaps;
	List<Integer[]> maxhs;
	List<Integer[]> maxvs;

	public poolingLayer(layer prev) {
		//  weightMaps = new ArrayList<>();
		this.prev = prev;
		outputSize = (prev.outputSize/CNN.pooling_Size); // prev.outsize must be divisible!
		count++;
		pl = new ArrayList<>();
		deltas = new ArrayList<>();
		for (int i = 0; i < CNN.NumofPerceptronsPerLayer; i++) {
			perceptron p = new perceptron();
			pl.add(p);
			Double[] delta = new Double[outputSize * outputSize * Lab3.unitsPerPixel];
			for (int j = 0; j < delta.length; j++) {
				delta[j] = 0.0;
			}
			deltas.add(delta);
		}
		if(count < 2){
			next = new convLayer(this);
		}
		else{
			next = new fullyConnectedLayer(this, 300, false);
		}
	}

	@Override
	public void getOut(Vector<Vector<Double>> input) {
		this.input = input;
		maxhs = new ArrayList<>();
		maxvs = new ArrayList<>();
		int plateSize = prev.outputSize;

		outputs = new Vector<>();
		for (perceptron p : pl) {
			List<List<weight>> weightMap = new ArrayList<>();
			Vector<Double> plate = input.get(pl.indexOf(p));
			if (plate.size() % (CNN.pooling_Size * CNN.pooling_Size) != 0) {
				// pad 0
				plateSize = (int) (Math.ceil(1.0 * prev.outputSize / CNN.pooling_Size));
				for (int y = 0; y < prev.outputSize; y++) {
					for (int x = prev.outputSize; x < plateSize; x++) {
						plate.insertElementAt(new Double(0), y * prev.outputSize + x);
					}
				}
				for (int y = prev.outputSize; y < plateSize; y++) {
					for (int x = 0; x < plateSize; x++) {
						plate.insertElementAt(new Double(0), plate.size());
					}
				}
				outputSize = plateSize/CNN.pooling_Size;
			}
			Integer[] maxhs_perplate = new Integer[(plateSize / CNN.pooling_Size) * (plateSize / CNN.pooling_Size) * Lab3.unitsPerPixel];
			Integer[] maxvs_perplate = new Integer[(plateSize / CNN.pooling_Size) * (plateSize / CNN.pooling_Size) * Lab3.unitsPerPixel];
			Vector<Double> pool_maxtrix = new <Double>Vector();
			for (int c = 0; c < Lab3.unitsPerPixel; c++) {
				for (int j = 0; j < plateSize; j += CNN.pooling_Size) {
					for (int i = 0; i < plateSize; i += CNN.pooling_Size) {
						Double curMax = plate.get(i + j * plateSize + c * plateSize * plateSize) ;
						int maxh = i, maxv = j;
						for (int a = 0; a < CNN.pooling_Size; a++) {
							for (int b = 0; b < CNN.pooling_Size; b++) {
								Double cur_Val = plate.get(((i + a) + (b + j) * plateSize)*Lab3.unitsPerPixel + c);
								if (cur_Val > curMax) {
									maxh = i + a;
									maxv = b + j;
									curMax = cur_Val;
								}
								if ((a == CNN.pooling_Size - 1) && (b == CNN.pooling_Size - 1)) {
									maxhs_perplate[c  + (i / CNN.pooling_Size + j / CNN.pooling_Size * outputSize)*Lab3.unitsPerPixel] = maxh;
									maxvs_perplate[c  + (i / CNN.pooling_Size + j / CNN.pooling_Size * outputSize)*Lab3.unitsPerPixel] = maxv;
								}
							}
						}
						pool_maxtrix.add(curMax);
					}
				}
			}
			maxhs.add(maxhs_perplate);
			maxvs.add(maxvs_perplate);
			outputs.add(pool_maxtrix);

		}

		weightSums = new Vector<>();
		deepCopyweightSums();

		next.getOut(outputs);
	}

	@Override
	public void backProp() {
		for (int i = 0; i < maxhs.size(); i++) {
			Integer [] maxh = maxhs.get(i);
			Integer [] maxv = maxvs.get(i);
			Double [] delta = deltas.get(i);
			Double[] prev_delta = prev.deltas.get(i);
			for(int c = 0; c < Lab3.unitsPerPixel; c++){
				for (int k = 0 ; k < outputSize; k++){
					for (int j = 0; j < outputSize ; j++) {
						//    prev.deltas.get(i)[maxh[j] + maxv[j] * prev.outputSize];
						//System.out.println("maxh[j]: " + maxh[j] + " " + " maxv[j]: " +  maxv[j]);
						Integer cur_delta_idx  = c + (k * outputSize + j)*Lab3.unitsPerPixel;
						Double delta_element =  delta[cur_delta_idx];
						// scale the value by the number of cells in the pooling layer
						delta_element/= (outputSize * outputSize * Lab3.unitsPerPixel);
						prev_delta [(maxh[cur_delta_idx] + maxv[cur_delta_idx] * prev.outputSize)*Lab3.unitsPerPixel + c ]= delta_element;
						prev.deltas.set(i, prev_delta);
					}
				}
			}

		}
//        System.out.println("Pooling BackProp");
//        printdeltas();
		prev.backProp();
		clearDeltas();
		outputs.clear();
	}
}

class fullyConnectedLayer extends layer{
	Map <perceptron, weight[][][][]> weightmap;
	int numUnits;
	boolean last;
	public fullyConnectedLayer(layer prev, int numUnits, boolean last) {
		this.prev = prev;
		outputSize = 1;
		this.numUnits = numUnits;
		this.last = last;
		int unitsPerPixel  = 1;// mix the color output at the very end
		if(!last){
			unitsPerPixel = Lab3.unitsPerPixel;
		}

		int fanout =  Lab3.Category.values().length;
		if(last){
			fanout = 1;
		}

		pl = new ArrayList<>();
		deltas = new ArrayList<>();
		weightmap = new HashMap<>();

		for (int pidx = 0; pidx < numUnits; pidx++) {
			perceptron p = new perceptron();
			pl.add(p);
			weight[][][][] wgt = new weight[prev.pl.size()][prev.outputSize][prev.outputSize + 1][unitsPerPixel]; // including bias weight
			for (int i = 0; i < prev.pl.size(); i++) {
				// initialize delta matrix

				for (int ii = 0; ii < prev.outputSize; ii++) {
					for (int j = 0; j < prev.outputSize + 1; j++) {
						for (int k = 0; k < unitsPerPixel; k++) {
							wgt[i][ii][j][k] = new weight(prev.pl.size(), fanout);
							// bias at wgt[0][0][prev.outputSize][0] presumably!
						}
					}
				}
			}
			Double[] delta = new Double[outputSize * outputSize];
			for (int i = 0; i < delta.length; i++) {
				delta[i] = new Double(0);
			}
			deltas.add(delta);
			weightmap.put(p, wgt);
		}
		// System.out.println(deltas.size() + " delta size fully_conn");
		if(!last){
			activationFunctionForHUs = "LeakyRELU";
			next = new fullyConnectedLayer(this, Lab3.Category.values().length, true);
		}
		else{
			activationFunctionForOutputs = "Sigmoid";
		}
	}
	@Override
	public void getOut(Vector<Vector<Double>> input) {
//         System.out.println("fullyConnectedLayer with layerNum: " + numUnits);
		outputs = new Vector<>();
		this.input = input;
		int unitsPerPixel  = 1;// mix the color output at the first fullyconnected layer
		if(!last){
			unitsPerPixel = Lab3.unitsPerPixel;
		}
		for (perceptron p : pl) {
			weight [][][][] wgt = weightmap.get(p);
			Vector<Double> sum_v = new Vector<>();
			Double sum = 0.0;
			for(int plateidx = 0; plateidx < prev.pl.size(); plateidx++){
				Vector<Double> curPlate = prev.outputs.get(plateidx);
				for(int c = 0 ; c < unitsPerPixel; c++){
					for(int x = 0; x < prev.outputSize; x ++){
						for (int y = 0; y < prev.outputSize; y ++){
							int idx = c + Lab3.unitsPerPixel  * (x * prev.outputSize + y);
							Double value = curPlate.get(idx);
							Double weight = wgt[plateidx][x][y][c].weight;
							sum +=  weight * value ;
						}
					}
				}
			}
			// bias
			sum += wgt[0][0][prev.outputSize][0].weight;
			sum_v.add(sum);
			outputs.add(sum_v);
		}

		weightSums = new Vector<>();
		deepCopyweightSums();
		activateLayer();
		if(!last){
			next.getOut(outputs);
		}
	}
	@Override
	public void backProp() {

		int unitsPerPixel = 1;// mix the color output at the very end
		if (!last) {
			unitsPerPixel = Lab3.unitsPerPixel;
			derivativeDeltas();
		}
		for (int i = 0; i < deltas.size(); i++) {
			weight[][][][] wgt = weightmap.get(pl.get(i));
			Double[] delta = deltas.get(i);
			for (int plateidx = 0; plateidx < prev.pl.size(); plateidx++) {
				Double[] prev_delta = prev.deltas.get(plateidx);
				for (int c = 0; c < unitsPerPixel; c++) {
					for (int x = 0; x < prev.outputSize; x++) {
						for (int y = 0; y < prev.outputSize; y++) {
							int idx = c + (x * prev.outputSize + y) * unitsPerPixel;
							Double weight = wgt[plateidx][x][y][c].weight;
							prev_delta[idx] += delta[0] * weight;
							Double input = prev.outputs.get(plateidx).get(idx);
							wgt[plateidx][x][y][c].weight += Lab3.eta * delta[0] * input;
						}
					}
				}
				prev.deltas.set(plateidx, prev_delta);
			}
			// update bias weight
			weight bias =  wgt[0][0][prev.outputSize][0];
			bias.weight += Lab3.eta * delta[0] * (1);
		}

//        System.out.println("fullyconnected backprop: ");
//        printdeltas();
		prev.backProp();
		clearDeltas();
		outputs.clear();
	}
	public Map <perceptron, weight[][][][]> getWeightmap (){
		return weightmap;
	}

}

class perceptron {

	boolean dropOut;
	public perceptron() {

	}


}
class weight{
	Double weight;

	public weight(int fanin, int fanout) {
		double range = Math.max(Double.MIN_VALUE, 1.0 / Math.sqrt(1.0 * (fanin + fanout)));
		weight = /*1.0;*/ (2.0 * random() - 1.0) * range;
	}

	public weight() {
		weight = Double.MIN_VALUE;
	}
}
class grid{
	Double delta;
	public grid(){
		delta = 0.0;
	}
}
/**
 * Referenced from
 * http://stackoverflow.com/questions/14677993/how-to-create-a-hashmap-with-two-keys-key-pair-value
 */
class key {

	private final perceptron x;
	private final perceptron y;

	public key(perceptron x, perceptron y) {
		this.x = x;
		this.y = y;
	}
	public key(perceptron y) {
		this.y = y;
		x = null;
	}



	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (!(o instanceof key)) return false;
		key key = (key) o;
		if(x != null)
			return (x == key.x && y == key.y) || (y == key.x && x == key.y);

		return y == key.y;
	}

	@Override
	public int hashCode() {
		int result = y.hashCode();
		if (x == null){
			return result* result*  31;
		}
		result =  31* result*result + (x.hashCode())*(x.hashCode());
		return result;
	}

}
