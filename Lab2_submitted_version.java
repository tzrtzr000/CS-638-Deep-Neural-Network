//Name: Hongyi Wang, Zirui Tao
package lab2;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

/**
 * Created by Zirui Tao on 2/12/2017.
 */
public class Lab2 {
	static int batchNum = 1;
	static int acidNum = 21;
	static int padNum = 8;
	static int windowSize = 2 * padNum + 1;
	static final int EPOCH_NUM = 1000;
	// static final int RAW_INPUT_LAYER_NUM = windowSize;
	static final int input_type_Nums = 21;
	static final int INPUT_LAYER_NUM = windowSize;
	static final int HIDDEN_LAYER_NUM = 12;
	static final int OUTPUT_LAYER_NUM = 3;
	static int ENSEMBLE_ANN_NUM = 1;
	static double beta = 0.01;
	/*
	 * static List<netPerceptron> input_layer; static List<netPerceptron>
	 * hidden_layer; static List<netPerceptron> output_layer;
	 */
	static double alpha = 0.1;
	static double decay = 10 * alpha / EPOCH_NUM;
	static double limit = 0.58;
	static int conut = 0;
	static double weightdecay = 0;

	public static void main(String[] args) throws IOException {
		String train_FileName = args[1];
		// String tune_FileName = args[1];
		List<Strian> train = ParseFile(train_FileName);
		List<Strian> removeList = new ArrayList<>();
		List<Strian> tune = new ArrayList<Strian>();
		List<Strian> test = new ArrayList<Strian>();

		// split index into train , tune and test set
		for (int i = 5; i < train.size(); i++) {
			if (i % 5 == 0) {
				tune.add(train.get(i));
				removeList.add(train.get(i));
				if (i + 1 < train.size()) {
					test.add(train.get(i + 1));
					removeList.add(train.get(i + 1));
				}
			}
		}

		// remove overlaps
		for (Strian strian : removeList) {
			train.remove(strian);
		}

		List<ANN> nl = new ArrayList<>();

		for (int i = 0; i < ENSEMBLE_ANN_NUM; i++) {
			ANN network1 = new ANN();
			/*
			 * network1.dropout(2); network1.dropout(1);
			 */
			nl.add(network1);
		}
		int count = 0;
		final double origin_alpha = alpha;
		double prev_tune_accuracy = 0;
		int batch_size = train.size() / ENSEMBLE_ANN_NUM;
		int i = 0;

		for (int y = 0; y < 1; y++) {
			alpha = 0.01;
			for (int x = 0; x < 1; x++) {
				weightdecay = 0;
				for (int v = 0; v < 1; v++) {
					ENSEMBLE_ANN_NUM = 1;
					for (int u = 0; u < 10; u++) {
						decay = 0.0001*u+0.0001;

						for (int z = 0; z < 10; z++) {
							for (i = 0; i < EPOCH_NUM; i++) {
								// int dropN = i/drop_size;
								// set the appropriate learning rate: decay as
								// time goes
								alpha = alpha * 1 / (1 + decay * i);
								// shuffle the order of list
								Collections.shuffle(train);
								for (ANN ann : nl) {
									// train
									int batch_num = nl.indexOf(ann);
									if (batch_num != nl.size() - 1) {
										List<Strian> subset = new ArrayList<Strian>(
												train.subList(batch_size * batch_num, (batch_num + 1) * batch_size));
										ann.training(train);
									} else {
										List<Strian> subset = new ArrayList<Strian>(
												train.subList(batch_size * batch_num, train.size()));
										ann.training(train);
									}
								}

								// scale weights
								for (ANN ann : nl) {
									ann.scale(1 - ann.dropout_rates);
								}

								double tune_accuracy = calculate_accuracy(tune, nl,false);

								// tune
								if (tune_accuracy > 0.65) {
									//System.out.println("tune_accuracy: " + tune_accuracy);
									break;
								}
								if ((tune_accuracy < prev_tune_accuracy) && (tune_accuracy > limit)) {
									if (count > 2) {
									//	System.out.println("tune_accuracy: " + tune_accuracy);
										break;
									} else {
										count++;
									}
								}

								prev_tune_accuracy = tune_accuracy;
								//System.out.println("tune_accuracy: " + tune_accuracy);
								// reset the learning rate
								alpha = origin_alpha;
								// reset the all the ANNs in nl
								for (ANN ann : nl) {
									ann.scale(1 / (1 - ann.dropout_rates));
								}
								

							}

							// scale weights
							/*
							 * for (ANN ann : nl) { ann.scale(1-
							 * ann.dropout_rates); }
							 */

							Double test_accuracy = calculate_accuracy(test, nl,true);
							//System.out.println("test accuracy: " + test_accuracy + " with count: " + count);
							// FileWriter outfile = new FileWriter(new
							// File("lab2_data.csv"), true);
						//	FileWriter outfile = new FileWriter(new File("alphadecay.csv"), true);
//							outfile.write(test_accuracy.toString() + ",");
//							outfile.write((Integer.toString(i+1)) + ",");
//							outfile.write(" " + ENSEMBLE_ANN_NUM + ",");
//							outfile.write(" " + HIDDEN_LAYER_NUM + ",");
//							outfile.write(" " + decay + ",");
//							outfile.write(" " + beta + ",");
//							outfile.write(" " + EPOCH_NUM + ",");
//							outfile.write(" " + limit + ",");
//					
//							outfile.write(" weightdecay " + weightdecay + ",");
//							outfile.write(" " + alpha + "\n");
//							outfile.close();
						}
					}
				}
			}
		}
	}


	private static void printWeights(List<ANN> nl) {
		for (ANN ann : nl) {
			System.out.println("ANN_NUM: " + nl.indexOf(ann));
			for (netPerceptron p : ann.output_layer) {
				System.out.println(Arrays.toString(p.weights));
				System.out.println("............................");
			}
			System.out.println("---------------------------");
		}
	}

	protected static double calculate_accuracy(List<Strian> test, List<ANN> nl, boolean testornot) {
		int correct = 0;
		int start = 0, end = windowSize - 1;
		int n = 0;
		// System.out.println("test set size: " + test.size());
		for (Strian s : test) {
			n += s.ps.size();
		}
		for (Strian s : test) {
			List<dfvec> raw_inputs = convertToOneHotFvec(s);
			start = 0;
			end = windowSize - 1;
			while (end < raw_inputs.size()) {
				char label = raw_inputs.get((start + end) / 2).label;
				List<Character> voteoutput = new ArrayList<Character>();
				for (ANN ann : nl) {
					List<dfvec> in = raw_inputs.subList(start, end + 1);
					ann.Calculate_Output(in);
					int maxidx = 0;
					double max = 0;
					for (netPerceptron p : ann.output_layer) {
						if (p.output > max) {
							maxidx = ann.output_layer.indexOf(p);
							max = p.output;
						}
					}
					char output = ' ';
					if (maxidx == 0) {
						output = '_';
					} else if (maxidx == 1) {
						output = 'e';
					} else if (maxidx == 2) {
						output = 'h';
					} else {
						System.exit(11);
					}
					if(testornot)
					 System.out.println(output);
					voteoutput.add(output);
					ann.clear_input_output_delta();
				}
				// ensemble final output
				// System.out.print( " voteoutput.size(): " +
				// (voteoutput.size()));
				char enout = majority(voteoutput);
				// if(print){
				// System.out.println( " enout: " + enout + "label :" + label);
				// }
				voteoutput.clear();
				// System.out.println("enout: " + enout + "label: " + label);
				if (enout == label) {
					correct++;
				} else {
				}
		
				start++;
				end++;
			}

		}
		return correct * 1.0 / n;
	}

	private static char majority(List<Character> l) {
		Map<Character, Integer> map = new HashMap<>();
		for (Character c : l) {
			Integer val = map.get(c);
			map.put(c, val == null ? 1 : val + 1);
		}

		Map.Entry<Character, Integer> max = null;
		for (Map.Entry<Character, Integer> e : map.entrySet()) {
			if (max == null || e.getValue() > max.getValue())
				max = e;
		}

		return max.getKey().charValue();
	}

	protected static List<dfvec> convertToOneHotFvec(Strian sample) {
		List<dfvec> dfvecs = new ArrayList<dfvec>();
		// List<dfvec> updateset = new ArrayList<dfvec>();

		// pad first padNum: 8

		for (int i = 0; i < padNum; i++) {
			dfvec solvent = new dfvec('0', 21, "S");
			dfvecs.add(solvent);
		}
		char label = ' ';
		for (pair p : sample.ps) {
			char acid_c = p.acid.charAt(0);

			dfvec curFeatureVector = new dfvec(acid_c, acidNum, p.label);
			dfvecs.add(curFeatureVector);
		}

		// pad last padNum: 8
		for (int i = 0; i < padNum; i++) {
			dfvec solvent = new dfvec('0', 21, "S");
			dfvecs.add(solvent);
		}

		return dfvecs;
	}

	private static List<Strian> ParseFile(String fileName) throws IOException {
		if (fileName == null || fileName.equals("")) {
			System.out.println("file name is null or empty");
			System.exit(10);
		}

		String[] lines = null;
		// Files.lines(Paths.get(fileName), StandardCharsets.UTF_8);
		List<Strian> samples = new ArrayList<Strian>();
		try {
			Stream<String> s = Files.lines(Paths.get(fileName));
			// convert into String arrays
			lines = s.toArray(size -> new String[size]);
			List<String> l = Arrays.asList(lines);
			samples = extractStrains(l);

			// System.out.println(samples);

		} catch (IOException e) {
			e.printStackTrace();
		}
		return samples;
	}

	private static List<Strian> extractStrains(List<String> raw_input) {
		// split according to "<>"
		List<Strian> samples = new ArrayList<>();
		// pair p = null;
		List<String> acids = new ArrayList<String>();
		List<String> labels = new ArrayList<String>();
		for (String s : raw_input) {
			if (s.trim().startsWith("#") || s.equals("")) {
				continue;
			}
			if (s.trim().equals("<>")) {
				// crate new pair by clearing the previous elements in acids
				// System.out.println("<>" );
				if (!acids.isEmpty() && !labels.isEmpty()) {
					Strian cursample = new Strian(acids, labels);
					samples.add(cursample);
					acids.clear();
					labels.clear();
				}

			} else if (s.trim().equals("end") || s.trim().equals("<end>")) {
				continue;
			} else {
				String[] fields = s.trim().split(" ");
				if (fields.length != 2) {
					System.out.println("splied line size is not 2 (acid + label)");
					System.exit(2);
				}
				if (fields[0].equals("")) {
					System.out.println("fields[0] = \"\",missing acid letter in this line ");
					System.exit(3);
				}
				if (fields[1].equals("")) {
					System.out.println("fields[1] = \"\",missing label in this line ");
					System.exit(4);
				}

				acids.add(fields[0]);
				labels.add(fields[1]);
				// System.out.println(fields[0] + " " + fields[1]);
			}

		}
		return samples;
	}
}

class ANN extends Lab2 {
	// static final double beta = 0.9;
	static final double dropout_rates = 0;// (1.0/3);
	List<netPerceptron> input_layer;
	List<netPerceptron> hidden_layer;
	List<netPerceptron> output_layer;
	Map<netPerceptron, Double[][]> droppedWeights;

	public ANN() {
		droppedWeights = new HashMap<>();
		// build network
		// forward
		input_layer = new ArrayList<netPerceptron>();
		for (int i = 0; i < INPUT_LAYER_NUM; i++) {
			netPerceptron p = new netPerceptron(1, null, input_type_Nums);
			input_layer.add(p);
		}

		hidden_layer = new ArrayList<netPerceptron>();
		for (int i = 0; i < HIDDEN_LAYER_NUM; i++) {
			netPerceptron p = new netPerceptron(2, input_layer, null);
			hidden_layer.add(p);
		}

		for (netPerceptron perceptron : input_layer) {
			perceptron.nextLayer = hidden_layer;
		}
		for (netPerceptron perceptron : hidden_layer) {
			perceptron.prevLayer = input_layer;
		}

		output_layer = new ArrayList<netPerceptron>();
		for (int i = 0; i < OUTPUT_LAYER_NUM; i++) {
			netPerceptron p = new netPerceptron(3, hidden_layer);
			output_layer.add(p);
		}

		for (netPerceptron perceptron : output_layer) {
			perceptron.prevLayer = hidden_layer;
		}
		for (netPerceptron perceptron : hidden_layer) {
			perceptron.nextLayer = output_layer;
		}

	}

	public void dropout(int layer) {
		if (layer != 1 && layer != 2) {
			System.out.println("layerNum: " + layer);
			System.exit(12);
		}

		// Randomly Drop out Perceptron on hidden layer
		Random r = new Random();
		// int hiddenDropOutNum = r.nextInt(HIDDEN_LAYER_NUM/2);
		if (layer == 1) {
			int dropNum = (int) Math.floor(HIDDEN_LAYER_NUM * dropout_rates);
			for (int i = 0; i < dropNum; i++) {
				int target = r.nextInt(HIDDEN_LAYER_NUM);
				// reset the weight for the target perceptron
				netPerceptron p = hidden_layer.get(target);
				// get all of the weights that connect to p
				Double[][] connected = new Double[2][];
				connected[0] = new Double[INPUT_LAYER_NUM];
				for (int j = 0; j < INPUT_LAYER_NUM; j++) {
					connected[0][j] = p.weights[j];
				}
				// System.arraycopy(p.weights, 0 , connected[0],0,
				// p.weights.length);
				connected[1] = new Double[p.nextLayer.size()];
				for (netPerceptron parent : p.nextLayer) {
					connected[1][p.nextLayer.indexOf(parent)] = parent.weights[hidden_layer.indexOf(p)];
				}
				if (droppedWeights.containsKey(p)) {
					// try again
					i--;
					continue;
				} else {
					// store connected weights to the map
					droppedWeights.put(p, connected);
				}
				// reset backward
				for (int j = 0; j < p.weights.length; j++) {
					p.weights[j] = 0;
				}
				// reset forward
				for (netPerceptron parent : p.nextLayer) {
					parent.weights[hidden_layer.indexOf(p)] = 0;
				}
			}
		}

		else {
			// Randomly Dropout Perceptron on input layer
			for (int i = 0; i < (int) Math.floor(INPUT_LAYER_NUM * dropout_rates); i++) {
				int target = r.nextInt(INPUT_LAYER_NUM);
				// reset the weight for the target perceptron
				netPerceptron p = input_layer.get(target);
				Double[][] connected = new Double[2][];
				connected[0] = new Double[input_type_Nums];
				// System.out.println(p.weights.length + " " +
				// connected[0].length);
				for (int j = 0; j < connected[0].length; j++) {
					connected[0][j] = p.weights[j];
				}
				// System.arraycopy(p.weights, 0 , connected[0],0,
				// p.weights.length);
				connected[1] = new Double[p.nextLayer.size()];
				for (netPerceptron parent : p.nextLayer) {
					int idx = hidden_layer.indexOf(parent);
					connected[1][p.nextLayer.indexOf(parent)] = parent.weights[idx];
				}
				if (droppedWeights.containsKey(p)) {
					// try again
					i--;
					continue;
				} else {
					// store connected weights to the map
					droppedWeights.put(p, connected);
				}

				// reset backward
				for (int j = 0; j < p.weights.length; j++) {
					p.weights[j] = 0;
				}
				// reset forward
				for (netPerceptron parent : p.nextLayer) {
					parent.weights[input_layer.indexOf(p)] = 0;
				}
			}
		}

	}

	public void training(List<Strian> train) {
		for (Strian strian : train) {
			// randomly drop out perceptron:
			dropout(2);
			dropout(1);
			List<dfvec> raw_inputs = convertToOneHotFvec(strian);
			int max_batchNum = raw_inputs.size() - (windowSize - 1);
			int real_batchNum = Math.min(batchNum, max_batchNum);

			if (real_batchNum < batchNum) {
				System.out.println("batchNum greater than maximum possible number of updates in "
						+ "in the current strain, so make the batchNum as "
						+ " the size of total number of movings for the input window, which the maximum number of "
						+ "batchNum for each strain (protein sequence) can take ");
				batchNum = real_batchNum;
			}
			// if(raw_inputs.isEmpty()) System.out.println("raw_inputs empty");
			List<List<dfvec>> batch = new ArrayList<>();
			int start = 0, end = windowSize - 1;

			while (end < raw_inputs.size()) {
				if (start % real_batchNum == 0 && start != 0) {
					update(batch);
					batch.clear();
				}
				List<dfvec> sample = raw_inputs.subList(start, end + 1);
				batch.add(sample);
				start++;
				end++;
			}
			// restore the perceptron
			recoverWeights();
		}
	}

	public void update(List<List<dfvec>> samples) {

		if (samples == null || samples.size() == 0) {
			System.out.println("empty samples list");
			System.exit(10);
		}
		List<Character> labels = Arrays.asList('_', 'e', 'h');

		List<dfvec> input_avg = new ArrayList<>(samples.get(0));

		// forward
		double[] errs = new double[OUTPUT_LAYER_NUM];
		for (List<dfvec> sample : samples) {
			if (samples.indexOf(sample) != 0) {
				for (dfvec f : input_avg) {
					f.add(sample.get(input_avg.indexOf(f)));
				}
			}
			Calculate_Output(sample);
			// calculate and update the errs
			// get the label: teacher vector
			double[] teacher = new double[OUTPUT_LAYER_NUM];
			teacher[labels.indexOf(sample.get((windowSize + 1) / 2 - 1).label)] = 1.0;

			for (int i = 0; i < errs.length; i++) {
				netPerceptron p = output_layer.get(i);
				errs[i] += teacher[i] - p.output;
			}
		}
		// average the sum to get the avg input (for batch update)
		/*
		 * for (dfvec f : input_avg) { for (int i = 0; i < f.features.length;
		 * i++) { f.features[i]/=(samples.size()); } }
		 */
		// System.out.println(input_avg);
		/*
		 * for (int i = 0; i < errs.length; i++) { System.out.print(errs[i] +
		 * " "); }
		 */

		// backprop delta
		/*
		 * if((conut < 100000)){ System.out.println(errs[0] + " " +
		 * output_layer.get(0).sigmoidP(output_layer.get(0).output)); conut++; }
		 */
		for (netPerceptron op : output_layer) {
			op.delta = op.sigmoidP(op.output) * errs[output_layer.indexOf(op)];
			// update next layer delta
			for (int i = 0; i < op.weights.length - 1; i++) {
				netPerceptron child = op.prevLayer.get(i);
				child.delta += child.ReLUP(child.netinput) * op.weights[i] * op.delta;
				// update output layer weights
				op.weights[i] += alpha * op.delta * child.output;
				// momentum
				op.weights[i] += beta * alpha * op.prevdelta * child.prevoutput;
				//Weight decay

				op.weights[i] -= weightdecay*alpha*op.weights[i];
			}
			// bias
			op.weights[op.weights.length - 1] += alpha * op.delta * (-1);
			op.weights[op.weights.length - 1] += beta * alpha * op.prevdelta * (-1);
			//Weight decay
			op.weights[op.weights.length - 1] -= weightdecay*alpha*op.weights[op.weights.length - 1];
		}

		for (netPerceptron hp : hidden_layer) {

			for (int i = 0; i < hp.weights.length - 1; i++) {
				netPerceptron child = hp.prevLayer.get(i);
				child.delta += hp.weights[i] * hp.delta;
				// update weights
				hp.weights[i] += alpha * hp.delta * child.output;
				// momentum
				hp.weights[i] += beta * alpha * hp.prevdelta * child.prevoutput;
				//Weight decay
				hp.weights[i] -= weightdecay*alpha*hp.weights[i];
			}

			// bias
			hp.weights[hp.weights.length - 1] += alpha * hp.delta * (-1);
			hp.weights[hp.weights.length - 1] -= weightdecay*alpha*hp.weights[hp.weights.length - 1];
			// momentum
			hp.weights[hp.weights.length - 1] += beta * alpha * hp.prevdelta * (-1);
		}

		for (netPerceptron ip : input_layer) {
			int index = input_layer.indexOf(ip);
			dfvec input = input_avg.get(index);
			for (int i = 0; i < ip.weights.length; i++) {
				ip.weights[i] += alpha * ip.delta * input.features[i];
				// momentum
				ip.weights[i] += beta * alpha * ip.prevdelta * input.features[i];
				//Weight decay
				ip.weights[i] -= weightdecay*alpha*ip.weights[i];
				
				
			}
		}

		/*
		 * if(conut < 71676){
		 * 
		 * netPerceptron p = output_layer.get(0); System.out.println("delta: " +
		 * p.delta + " " + Arrays.toString(p.weights) + " " + conut++);
		 * System.out.println("-----------------"); }
		 */

		clear_input_output_delta();

	}

	public void Calculate_Output(List<dfvec> sample) {
		double[] output = new double[OUTPUT_LAYER_NUM];
		// System.out.println("sample size: " + sample.size() + "input_layer: "
		// + input_layer.size());
		// input layer
		for (dfvec featureVec : sample) {
			int i = sample.indexOf(featureVec);
			netPerceptron p = input_layer.get(i);
			p.CalculateWeightedSum(featureVec.features);
			// input layer: output = input
			p.output = p.netinput;
		}

		// hidden layer
		for (netPerceptron hp : hidden_layer) {
			for (netPerceptron child : hp.prevLayer) {
				hp.netinput += child.output * hp.weights[hp.prevLayer.indexOf(child)];
			}
			hp.output = hp.ReLU(hp.netinput);
		}

		// output
		for (netPerceptron op : output_layer) {
			for (netPerceptron child : hidden_layer) {
				op.netinput += child.output * op.weights[hidden_layer.indexOf(child)];
			}
			op.output = op.sigmoid(op.netinput);
		}

	}

	public void clear_input_output_delta() {
		// clear input and output and delta for each perceptron
		for (netPerceptron p : input_layer) {
			p.prevdelta = p.delta;
			p.prevoutput = p.output;
			p.output = 0;
			p.netinput = 0;
			p.delta = 0;
		}
		for (netPerceptron p : hidden_layer) {
			p.prevdelta = p.delta;
			p.prevoutput = p.output;
			p.output = 0;
			p.netinput = 0;
			p.delta = 0;
		}
		for (netPerceptron p : output_layer) {
			p.prevdelta = p.delta;
			p.prevoutput = p.output;
			p.output = 0;
			p.netinput = 0;
			p.delta = 0;
		}
	}

	public void recoverWeights() {
		for (netPerceptron key : droppedWeights.keySet()) {
			Double[][] weights = droppedWeights.get(key);
			// System.arraycopy(weights[0],0,key.weights,0,weights[0].length);
			for (int i = 0; i < weights[0].length; i++) {
				key.weights[i] = weights[0][i];
			}
			for (netPerceptron parent : key.nextLayer) {
				parent.weights[parent.prevLayer.indexOf(key)] = weights[1][key.nextLayer.indexOf(parent)];
			}
		}
		droppedWeights.clear();
	}

	public void scale(double factor) {
		for (netPerceptron ip : input_layer) {
			for (int i = 0; i < ip.weights.length; i++) {
				ip.weights[i] *= factor;
			}
		}
		for (netPerceptron hp : hidden_layer) {
			for (int i = 0; i < hp.weights.length; i++) {
				hp.weights[i] *= factor;
			}
		}
	}
}

class Strian {
	static int ID = 1;
	List<pair> ps;

	public Strian(List<String> acids, List<String> labels) {
		ps = new ArrayList<pair>();
		if (acids.size() != labels.size()) {
			System.out.println("acids list and labels are in different size, " + "acids.size(): " + acids.size()
					+ "labels.size(): " + labels.size());
			System.exit(1);
		}

		for (int i = 0; i < acids.size(); i++) {
			pair p = new pair(acids.get(i), labels.get(i));
			ps.add(p);
		}
		ID++;

	}

	public Strian(List<pair> ps) {
		this.ps = ps;
	}

	@Override
	public String toString() {
		String out = "ID: " + ID;
		for (pair p : ps) {
			out += p.toString();
		}
		return out;
	}
}

class pair {
	// static int count = 10;
	String acid;
	String label;

	public pair(String acid, String label) {
		this.acid = acid;
		this.label = label;
	}

	@Override
	public String toString() {
		return acid + " " + label + " " + "\n";
	}
}

class dfvec {
	static HashMap<Character, Integer> dic = new HashMap<>();
	static int index = 0;
	int flen;
	double features[];
	char label;

	public dfvec(char input, int flen, String label) {
		// check if acid type exists
		int indexOfOne = 0;
		if (!dic.containsKey(input)) {
			dic.put(input, index);
			index++;
		}
		indexOfOne = dic.get(input);
		features = new double[flen];
		features[indexOfOne] = 1.0;
		this.flen = flen;
		label = label.trim();
		if ((label == null) || (label.length() != 1)) {
			System.out.println("string label not a single char: " + label);
			System.exit(8);
		}
		this.label = label.charAt(0);
	}

	public dfvec add(dfvec other) {
		for (int i = 0; i < features.length; i++) {
			features[i] += other.features[i];
		}
		return this;
	}

	@Override
	public String toString() {
		return "dfvec{" + "flen=" + flen + ", features=" + Arrays.toString(features) + '}';
	}
}

class data {
	List<dfvec> input;
	char label;

	public data(List<dfvec> input, char label) {
		this.input = input;
		this.label = label;
	}
}

class netPerceptron extends Lab2 {
	// static double alpha = 0.01;
	// static double beta = 0.1;
	double[] weights;
	List<netPerceptron> nextLayer;
	List<netPerceptron> prevLayer;
	int layerNum;
	double netinput;
	double output;
	double delta;
	double prevdelta;
	double prevoutput;

	// layerNum 1: raw_input layer's perceptron constructor: without prevLayer
	public netPerceptron(int layerNum, List<netPerceptron> nextLayer, int input_Num) {
		this.layerNum = layerNum;
		if (layerNum == 1) {
			this.nextLayer = nextLayer;
		} else {
			System.out.println("layerNum number error, 1: input 2: hidden(ReLu) 3: output(sigmoid)");
			System.out.println("expected layerNum: 1, + actual layerNum: " + layerNum);
			System.exit(7);
		}

		// initialize weights
		Random r = new Random();
		weights = new double[input_Num];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = (2 * r.nextDouble() - 1) * 4 / (Math.sqrt(6 * (input_type_Nums + HIDDEN_LAYER_NUM)));
		}

		// generate random weight value for bias
		/*
		 * Random r = new Random(); weights[input.features.length] =
		 * r.nextDouble();
		 */

	}

	// layerNum 4: output(sigmoid as activation function) layer's perception
	// constructor: without nextLayer
	public netPerceptron(int layerNum, List<netPerceptron> prevLayer) {
		// generate using random Generators
		if (layerNum == 3) {
			this.prevLayer = prevLayer;
		} else {
			System.out.println("layerNum number error, 1:input 2: hidden(ReLu) 3: output(sigmoid)");
			System.out.println("expected layerNum: 3, + actual layerNum" + layerNum);
			System.exit(7);
		}

		// initialize weights
		Random r = new Random();
		weights = new double[prevLayer.size() + 1];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = (2 * r.nextDouble() - 1) * (4 / (Math.sqrt(6 * HIDDEN_LAYER_NUM)));
		}
	}

	public netPerceptron(int layerNum, List<netPerceptron> prevLayer, List<netPerceptron> nextLayer) {
		if ((layerNum == 2)) {
			this.prevLayer = prevLayer;
			this.nextLayer = nextLayer;
		} else {
			System.out.println("layerNum number error, 1:input 2: hidden(ReLu) 3: output(sigmoid)");
			System.out.println("expected layerNum: 2, + actual layerNum" + layerNum);
			System.exit(7);
		}

		this.layerNum = layerNum;
		// initialize weights : forward
		Random r = new Random();
		weights = new double[prevLayer.size() + 1];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = (2 * r.nextDouble() - 1) * (4 / Math.sqrt(6 * (INPUT_LAYER_NUM + OUTPUT_LAYER_NUM)));
		}

	}

	public void addnextLayers(List<netPerceptron> nextLayer) {
		this.nextLayer = nextLayer;
	}

	public void prevLayer(List<netPerceptron> prevLayer) {
		this.prevLayer = prevLayer;
	}

	public void CalculateWeightedSum(double[] input) {
		if (input.length != weights.length) {
			System.out.println("input and weights size does not match");
			System.out.println("input: " + input.length + "weight: " + weights.length);
			System.exit(8);
		}
		double sum = 0;
		for (int i = 0; i < input.length; i++) {
			sum += input[i] * weights[i];
		}

		// bias weight
		/* sum+= (-1)*weights[input.length ]; */
		netinput = sum;

		// return sum;
	}

	public double sigmoid(double x) {
		return 1 / (1 + (Math.exp(-x)));
	}

	public double sigmoidP(double x) {
		return x * (1 - x);
	}

	public double ReLU(double x) {

		return Math.max(0, x);
	}

	public double ReLUP(double x) {
		// TODO
		return x > 0 ? 1 : 0;
	}

}
