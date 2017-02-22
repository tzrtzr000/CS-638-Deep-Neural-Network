import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/***
 * main program of the project
 */
public class Lab1 {
    static String l1,l2 = "";
    static int featureNum = 0;
    static String f_val1, f_val2="";
    /***
     * Main method of the project
     */
    public static void main(String[] args) {
        // write your code here
        final int EPOCHNUM = 1000;
        String trainIn = args[0];
        String tuneIn = args[1];
        String testIn = args[2];
        // parser
        System.out.println(trainIn + tuneIn + testIn);
        List<fvec> trainVecs = ParseFile(trainIn);
        List<fvec> tuneVecs = ParseFile(tuneIn);
        List<fvec> testVecs = ParseFile(testIn);

        // train
        // initialize perception(s)
        perceptron p = new perceptron(featureNum + 1);

        // train the feature vector using early stoping
        double prevTune = 0;
        for(int i = 0 ; i < EPOCHNUM ; i ++) {
            Collections.shuffle(trainVecs);
            // train
            train(trainVecs,p);
            // tune
            double accuracy_Tune = CalculateAccuracy(tuneVecs, p);
            // check if stoping criteria is met
            if(accuracy_Tune > 0.8 && accuracy_Tune < prevTune){
                break;
            }
            prevTune = accuracy_Tune;
        }

        // for test set, print the predicted label value, each line per sample, in the sample order as input
        printPredictedCategory(testVecs, p);

        // test
        double accuracy_Test = CalculateAccuracy(testVecs, p);
        System.out.println("TestAccuracy: " + accuracy_Test);

    }


    /***
     * Print the test set predicted labels
     * @param testVecs
     * @param p
     */
    private static void printPredictedCategory(List<fvec> testVecs, perceptron p) {
        final double thre = 0.5;
        int o = 0;
        for (fvec f : testVecs) {
            double out = f.getResult(p);
            o = out > thre ? 1 : 0;
            String printOut = o == 1? l1:l2;
            System.out.println(printOut);
        }
    }

    /***
     * Parse the input file and return an list of feature vector (fvec)
     * @param fileName
     * @return an list of feature vector
     */
    private static List<fvec> ParseFile(String fileName) {
        int sampleNum = 0;
        // train
        try {
            FileReader fr = new FileReader(fileName);
            BufferedReader bufreader = new BufferedReader(fr);
            String line;
            int lineCount = 1;
            List<fvec> fvecs = new ArrayList<>();
            while ((line = bufreader.readLine()) != null) {
                if (!line.startsWith("//") && !(line.equals(""))) {
                    if (lineCount == 1) {
                        try {
                            featureNum = Integer.parseInt(line);
                          //  System.out.println("featureNum: " + featureNum);
                        } catch (Exception e) {
                            System.out.println("First Line is not a number!");
                            System.exit(1);
                        }
                        lineCount++;
                        continue;
                    }
                    if(lineCount == 2){
                        String[] split = line.split(" ");
                        f_val2 = split[split.length - 1];
                        f_val1 = split[split.length - 2];
                    }

                    // System.out.println(line);
                    if (lineCount == featureNum + 2) {
                        l1 = (line);
                    }
                    if (lineCount == featureNum + 3) {
                        l2 = (line);
                    }
                    if (lineCount == featureNum + 4) {
                        sampleNum = Integer.parseInt(line);
                    }

                    if (lineCount > featureNum + 4) {
                        fvec f = createFeatureVector(line);
                        fvecs.add(f);
                    }

                    lineCount++;
                }

            }

            return fvecs;

        } catch (FileNotFoundException e) {
            System.out.println("File Not Found");
        } catch (IOException ex) {
            System.out.println("Error reading file '" + "'");
        }
            return null;
    }


    /**
     *  create a feature vector from a line of string
     * @param line
     * @return the created featire vector
     */
    private static fvec createFeatureVector(String line) {
        if(line == null || line == ""){
            System.out.println("Line is empty or Null!");
            System.exit(2);
        }

        String[] raw_elmts = line.split(" ");

        // assume formatted inputs:

        String id = raw_elmts [0];

        String [] IDandLabel = new String[2];

        // extract labels from split string array
        int k = 0;

        for (int j = 0; j < raw_elmts.length; j++) {
            if(!raw_elmts[j].equals("")){
                IDandLabel[k] = raw_elmts[j];
                k++;
                if(k >= 2){
                    break;
                    }
                }
            }


        int label = 0;
        if(IDandLabel[1].equals(l1)){

        }
        else if(IDandLabel[1].equals(l2)){
            label = 1;
        }
        else{
            System.out.println("input ID is not Catagorized !");
            System.exit(4);
        }

        List<String> elmts = new ArrayList<>();
        for (int i = 2; i < raw_elmts.length; i++) {
            if((raw_elmts[i].trim().equals(f_val1)) || (raw_elmts[i].trim().equals(f_val2))){
                elmts.add(raw_elmts[i].trim());
            }
        }

        String [] processed_Elmts = new String [featureNum];
        elmts.toArray(processed_Elmts);
        // remove non binary terms, if any
        int[] features = new int [processed_Elmts.length];
        for (int i = 0; i < features.length; i++) {
           // boolean isT = raw_elmts[i+2].trim()=="T";
            features[i] = (processed_Elmts[i].trim().equals(f_val1)? 1: 0);
        }


        return new fvec(features,label, id);
    }

    /***
     * Train the perceptron for the later prediction on test set
     * @param fveclist
     * @param p
     */
    public static void train(List<fvec> fveclist, perceptron p){

        if(fveclist == null || fveclist.isEmpty()){
            System.out.println("fveclist is null or empty!");
            System.exit(3);
        }

        double alpha = 0.001;
        int sampleSize = fveclist.size();
        int numUpdates =(int)Math.ceil((1.0*sampleSize)/1000);

        // batch update: 1000 units per update
        List<fvec> updateCandidates = new ArrayList<>();
        for(int i = 1; i <= numUpdates; i++) {
            updateCandidates.clear();
            if (i == numUpdates) {
                updateCandidates = fveclist.subList((numUpdates - 1) * 1000, sampleSize - 1);

            } else {
                updateCandidates = fveclist.subList((i - 1) * 1000, 1000 * i);
            }


            List<Double> errs = new ArrayList<>();
            for (fvec f : updateCandidates) {
                double o = f.getResult(p);
                errs.add(f.label - o);
            }

            // update the weights
            for (int k = 0; k < errs.size(); k++) {
                double err = errs.get(k);
                double output = updateCandidates.get(k).getResult(p);
                fvec cur = updateCandidates.get(k);
                for (int j = 0; j < p.weights.length - 1; j++) {
                    // apply gradient decent
                    double delta_w = alpha * cur.features[j] * err * output * (1 - output);
                    p.weights[j] += delta_w;
                }
                double delta_bias = alpha * (-1) * err * output * (1 - output);
                p.weights[p.weights.length - 1] += delta_bias;
            }
        }
    }

    /***
     * Calculate the accuracy from
     * @param fvecs
     * @param p
     * @return
     */
    private static double CalculateAccuracy(List<fvec> fvecs, perceptron p) {

        if(fvecs == null || fvecs.isEmpty()){
            System.out.println("fveclist is null or empty!");
            System.exit(3);
        }
        final double thre = 0.5;
        int score = 0;
        int o = 0;
        for (fvec f : fvecs) {
            double out = f.getResult(p);
            o = out > thre ? 1 : 0;
            if((f.label == o)){
                score++;
            }

        }
        return (1.0*score/(fvecs.size()));
    }
}

/***
 * class of feature vector
 */
class fvec {
    int flen;
    int features[];
    int label;
    String ID;
    public fvec (int[] features, int label, String id){
        ID = id;
        flen = features.length ;
        this.features = new int [features.length];
        this.label = label;
        java.lang.System.arraycopy(features, 0, this.features,0, flen);

    }
    public double getResult (perceptron p){
        if(features.length != flen){
            System.out.println("features length not matched with flen");
        }

        double net_sum = 0;
        for( int i = 0 ; i< flen;i ++){
            net_sum += features[i]* p.weights[i];
        }

        // bias: a special term whose "input" is always 1
        net_sum += (-1) * p.weights[flen];
        double output = 1/(1+Math.exp(-net_sum));
        return output;
    }

    /***
     * print the result of the feature vector
     * @return
     */
    @Override
    public String toString() {
       String out = ID + " ";
        out += Integer.toString(label) + " ";

        for (int i = 0; i < features.length; i++) {
        out += features[i]+ " ";
        }
        out += "\n";

        return out;
    }
}

/***
 *  The perceptron class
 */
class perceptron {

    double weights [];
    public perceptron (int weightNums){
        weights = new double [weightNums];
        // Initialize weight vector: using Random Number Generator
        Random rand = new Random();
        for (int i = 0 ; i <  weights.length; i++){
            //assign the weights from randGen
            double  wVal  = rand.nextDouble();
            weights[i]= wVal;
            weights[i]= wVal;
        }
}

    /***
     * print the result of the perceptron
     * @return
     */
    @Override
    public String toString() {
        return "perceptron{" +
                "weights=" + Arrays.toString(weights) +
                '}';
    }

}