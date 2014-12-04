//        Copyright [BKYoo]
//
//        Licensed under the Apache License, Version 2.0 (the "License");
//        you may not use this file except in compliance with the License.
//        You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//        Unless required by applicable law or agreed to in writing, software
//        distributed under the License is distributed on an "AS IS" BASIS,
//        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//        See the License for the specific language governing permissions and
//        limitations under the License.

package com.kse.bigdata.main;

import com.kse.bigdata.entity.Sequence;
import com.kse.bigdata.file.SequenceSampler;
import com.kse.bigdata.file.SourceFileMerger;
import org.apache.commons.math3.analysis.function.Log;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.*;
import java.math.BigDecimal;
import java.util.*;

/**
 * Created by user on 2014-12-02.
 * KSE526 Term Project
 */
public class Driver {

    public static class Map extends Mapper<LongWritable, Text, NullWritable, Text> {
        public static final String NUMBER_OF_NEAREAST_NEIGHBOR = "nnn";
        public static final String NORMALIZATION                = "normalize";
        public static final String INPUT_SEQUENCE               = "inputSeq";
//        public static final String EUCLIDEAN_DISTANCE_THRESHOLD = "eucDist";

//        private double euclideanDistThreshold = 0.0d;
        private boolean normalization         = false;
        private int NUMBER_OF_NEIGHBOR        = 100;

        private Sequence userInputSequence;

        private Text result = new Text();

        private LinkedList<Double> temp;
        //private LinkedList<Sequence> sequences = new LinkedList<>();
        private SortedSet<Sequence> sequences = new TreeSet<>();

        private HashMap<Integer, LinkedList<Double>> tempSequence = new HashMap<>();

        private StandardDeviation standardDeviation = new StandardDeviation();
        private Mean              mean              = new Mean();
        private EuclideanDistance euclideanDistance = new EuclideanDistance();

        @Override
        public void setup(Context context) throws IOException{
//            euclideanDistThreshold = 1.0d * context.getConfiguration().getInt(EUCLIDEAN_DISTANCE_THRESHOLD, 10);
            userInputSequence      = new Sequence(context.getConfiguration().get(INPUT_SEQUENCE, ""));
            NUMBER_OF_NEIGHBOR = context.getConfiguration().getInt(NUMBER_OF_NEAREAST_NEIGHBOR, 100);
            normalization          = context.getConfiguration().getBoolean(NORMALIZATION, false);
        }


        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
            String[] tokens = value.toString().split(",");
            double powerGeneration = Double.parseDouble(tokens[0]);
            int sourceFileId = Integer.parseInt(tokens[1]);

            if(tempSequence.containsKey(sourceFileId)) {
                tempSequence.get(sourceFileId).add(powerGeneration);

            } else {
                tempSequence.put(sourceFileId, new LinkedList<Double>());

            }

            temp = tempSequence.get(sourceFileId);

            if(temp.size() == Sequence.SIZE_OF_SEQUENCE){
                Sequence newSeq = new Sequence(temp);
                newSeq.setEuclideanDistance(calculateEuclideanDistance(newSeq, userInputSequence));
                addWordToSortedSet(newSeq);
                temp.removeFirst();
            }

        }

        private void addWordToSortedSet(Sequence newSeq){
            sequences.add(newSeq);

            if(sequences.size() > NUMBER_OF_NEIGHBOR)
                // last element has the smallest frequency among the sequences.
                sequences.remove(sequences.last());
        }

        @Override
        public void cleanup(Context context) throws InterruptedException, IOException{
            for(Sequence seq : sequences) {
                result.set(seq.toString());
                context.write(NullWritable.get(), result);
            }
        }

        private double calculateEuclideanDistance(Sequence seqA, Sequence seqB){
            if(normalization) {
                normalize(seqA);
                normalize(seqB);
                return euclideanDistance.compute(seqA.getNormHead(), seqB.getNormHead());

            } else {
                return euclideanDistance.compute(seqA.getHead(), seqB.getHead());
            }
        }

        private void normalize(Sequence targetSeq){
            double[] normHead = targetSeq.getNormHead();
            double[] head     = targetSeq.getHead();

            double avg = mean.evaluate(targetSeq.getHead());
            double std = standardDeviation.evaluate(targetSeq.getHead());

            for(int index = 0; index < Sequence.SIZE_OF_HEAD_SEQ; index++){
                normHead[index] = (head[index] - avg) / std;
            }
        }
    }

    public static class Combiner extends Reducer<NullWritable, Text, NullWritable, Text>{

        private SortedSet<Sequence> sequences = new TreeSet<>();
        private final int BUFFER_SIZE = 500;

        @Override
        protected void reduce(NullWritable ignore, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            for(Text value : values) {
                Sequence newSeq = new Sequence(value.toString());
                addWordToSortedSet(newSeq);

                if(newSeq.getEuclideanDistance() <= sequences.last().getEuclideanDistance())
                    context.write(NullWritable.get(), value);

            }
        }

        private void addWordToSortedSet(Sequence newSeq){
            sequences.add(newSeq);

            if(sequences.size() > BUFFER_SIZE)
                // last element has the smallest frequency among the sequences.
                sequences.remove(sequences.last());
        }
    }

    public static class Reduce extends Reducer<NullWritable, Text, Text, Text> {
        public static final String NUMBER_OF_NEAREAST_NEIGHBOR = "nnn";
        public static final String INPUT_SEQUENCE              = "inputSeq";

        private final boolean MEAN        = true;
        private final int DECIMAL_SCALE   = 2;
        private final int ROUNDING_METHOD = BigDecimal.ROUND_HALF_UP;

        // Constant for checking the size of sequences.
        protected int NUMBER_OF_NEIGHBOR = 0;

        // Variables for reducing the cost of creating instance.
        private Text tempKey = new Text();
        private Text tempValue = new Text();

        private Sequence userInputSequence;

        // Container of word for 100 most frequent sequences.
        private SortedSet<Sequence> sequences = new TreeSet<>();

        private Mean mean     = new Mean();
        private Log  log      = new Log();
        private Median median = new Median();

        @Override
        public void setup(Context context) throws IOException{
            // one for same user input
            NUMBER_OF_NEIGHBOR = context.getConfiguration().getInt(NUMBER_OF_NEAREAST_NEIGHBOR, 100) + 1;
            userInputSequence  = new Sequence(context.getConfiguration().get(INPUT_SEQUENCE, ""));
        }

        @Override
        protected void reduce(NullWritable ignore, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            for(Text value : values) {
                Sequence newSeq = new Sequence(value.toString());
                addWordToSortedSet(newSeq);
            }
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            double[] predictedValues = predictSequence();

            StringBuffer buf =  new StringBuffer();
            for(int index = 0; index < predictedValues.length; index++){
                buf.append(predictedValues[index]);

                if(index == (predictedValues.length - 1))
                    break;

                buf.append("-");
            }

            double MER = calculateMeanErrorRate(predictedValues, userInputSequence.getTail());
            double MAE = calculateMeanAbsoluteError(predictedValues, userInputSequence.getTail());

            tempKey.set(buf.toString());
            tempValue.set(String.valueOf(MER) + " " + String.valueOf(MAE));
            context.write(tempKey, tempValue);
        }

        private double[] predictSequence(){
            double[] predictionResult = new double[Sequence.SIZE_OF_TAIL_SEQ];
            ArrayList<double[]> tails = new ArrayList<>();

            double sumOfWeights = 0.0d;
            double[] tailOfSeq;
            double weight;

            int counter = 0;
            for(Sequence seq : this.sequences) {
                tailOfSeq = seq.getTail();

                if (MEAN) {
                    weight = calculateWeight(seq);
                    sumOfWeights += weight;

                    for (int index = 0; index < Sequence.SIZE_OF_TAIL_SEQ; index++) {
                        predictionResult[index] += weight * tailOfSeq[index];
                    }

                } else {
                    for (int index = 0; index < Sequence.SIZE_OF_TAIL_SEQ; index++) {
                        if(tails.get(index) == null)
                            tails.add(new double[this.sequences.size()]);

                        tails.get(index)[counter] = tailOfSeq[index];
                    }

                    counter++;
                }
            }

            for (int index = 0; index < Sequence.SIZE_OF_TAIL_SEQ; index++) {
                if(MEAN) {
                    predictionResult[index] = Precision.round((predictionResult[index] / sumOfWeights),
                            DECIMAL_SCALE, ROUNDING_METHOD);
                } else {
                    predictionResult[index] = median.evaluate(tails.get(index));
                }
            }

            return predictionResult;
        }

        private void addWordToSortedSet(Sequence newSeq){
            sequences.add(newSeq);

            if(sequences.size() > NUMBER_OF_NEIGHBOR)
                // last element has the smallest frequency among the sequences.
                sequences.remove(sequences.last());
        }

        private double calculateWeight(Sequence seq){
            double eucDist = seq.getEuclideanDistance();

            if(eucDist == 0)
                eucDist = 0.00001;

            return log.value(1.0d/eucDist);
        }

        private double calculateMeanErrorRate(double[] predictedSeq, double[] actualSeq){
            double meanOfActualSeq = mean.evaluate(actualSeq);
            double error = 0.0d;

            for(int index = 0; index < Sequence.SIZE_OF_TAIL_SEQ; index++)
                error += FastMath.abs(predictedSeq[index] - actualSeq[index]);

            return Precision.round(100 * error / (Sequence.SIZE_OF_TAIL_SEQ * meanOfActualSeq),
                    DECIMAL_SCALE, ROUNDING_METHOD);
        }

        private double calculateMeanAbsoluteError(double[] predictedSeq, double[] actualSeq){
            double error = 0.0d;
            for(int index = 0; index < Sequence.SIZE_OF_TAIL_SEQ; index++)
                error += FastMath.abs(predictedSeq[index] - actualSeq[index]);

            return Precision.round((error / Sequence.SIZE_OF_TAIL_SEQ), DECIMAL_SCALE, ROUNDING_METHOD);
        }

    }

    public static void main(String[] args) throws Exception {
        /**
         * Merge the source files into one.
         */
        //##################################################################################
        //##    Should change the directories of each file before executing the program   ##
        //##################################################################################
        String inputFileDirectory = "/media/bk/드라이브/BigData_Term_Project/Data";
        String resultFileDirectory = "/media/bk/드라이브/BigData_Term_Project/data.csv";
        File resultFile = new File(resultFileDirectory);
        if(!resultFile.exists())
            new SourceFileMerger(inputFileDirectory, resultFileDirectory).mergeFiles();

        if(false) {
            /**
             * Hadoop Operation.
             */
            Configuration conf = new Configuration();

            //Enable MapReduce intermediate compression as Snappy
            conf.setBoolean("mapred.compress.map.output", true);
            conf.set("mapred.map.output.compression.codec", "org.apache.hadoop.io.compress.SnappyCodec");

            //Enable Profiling
            //conf.setBoolean("mapred.task.profile", true);

            String testPath = null;
            String inputPath = null;
            String outputPath = null;

            int sampleSize = 1;
            ArrayList<String> results = new ArrayList<>();

            for (int index = 0; index < args.length; index++) {

                /**
                 * Mandatory command
                 */
                //Extract input path string from command line.
                if (args[index].equals("-in"))
                    inputPath = args[index + 1];

                //Extract output path string from command line.
                if (args[index].equals("-out"))
                    outputPath = args[index + 1];

                if (args[index].equals("-test"))
                    testPath = args[index + 1];

                /**
                 * Optional command
                 */
                //Euclidean distance threshold.
//            if(args[index].equals("-dist")) {
//                conf.setInt(Map.EUCLIDEAN_DISTANCE_THRESHOLD, Integer.valueOf(args[index + 1]));
//            }
                //Number of neighbor
                if (args[index].equals("-nn"))
                    conf.setInt(Reduce.NUMBER_OF_NEAREAST_NEIGHBOR, Integer.valueOf(args[index + 1]));

            //Normalization
            if(args[index].equals("-norm"))
                conf.setBoolean(Map.NORMALIZATION, true);

                //Test Sample Size
                if (args[index].equals("-s"))
                    sampleSize = Integer.valueOf(args[index + 1]);
            }

            String outputFileName = "part-r-00000";
            String finalOutputPath = "project/output/result";

            SequenceSampler sampler = new SequenceSampler(testPath, sampleSize);
            LinkedList<Sequence> testSequences = sampler.getRandomSample();

            for (Sequence seq : testSequences) {
                System.out.println(seq.getTailString());

                conf.set(Map.INPUT_SEQUENCE, seq.toString());

                Job job = new Job(conf);
                job.setJarByClass(Driver.class);
                job.setJobName("term-project-driver");

                job.setMapperClass(Map.class);
                job.setMapOutputKeyClass(NullWritable.class);
                job.setMapOutputValueClass(Text.class);

                //Set 1 for number of reduce task for keeping 100 most sequences in sorted set.
                job.setReducerClass(Reduce.class);
                job.setOutputKeyClass(Text.class);
                job.setOutputValueClass(Text.class);
                job.setNumReduceTasks(1);

                job.setInputFormatClass(TextInputFormat.class);
                job.setOutputFormatClass(TextOutputFormat.class);

                FileInputFormat.setInputPaths(job, new Path(inputPath));
                FileOutputFormat.setOutputPath(job, new Path(outputPath));

                job.waitForCompletion(true);

                try (FileSystem hdfs = FileSystem.get(new Configuration());) {

                    BufferedReader fileReader = new BufferedReader(new InputStreamReader(
                            hdfs.open(new Path(outputPath + "/" + outputFileName))));

                    String line;
                    while ((line = fileReader.readLine()) != null) {
                        results.add(seq.getTailString() + " " + line);
                    }

                    fileReader.close();

                    hdfs.delete(new Path(outputPath), true);

                } catch (IOException e) {
                    e.printStackTrace();
                    System.exit(1);
                }
            }

            try (FileSystem hdfs = FileSystem.get(new Configuration());) {

                Path file = new Path(finalOutputPath);
                if (hdfs.exists(file)) {
                    hdfs.delete(file, true);
                }

                OutputStream os = hdfs.create(file);
                PrintWriter printWriter = new PrintWriter(new OutputStreamWriter(os, "UTF-8"));

                double totalMER = 0.0d;
                double totalMAE = 0.0d;

                for (String result : results) {
                    String[] tokens = result.split("\\s+");

                    totalMER += Double.valueOf(tokens[2]);
                    totalMAE += Double.valueOf(tokens[3]);

                    String actualSeq = "A:" + tokens[0];
                    String predictedSeq = "P:" + tokens[1];
                    String errorOutput = " [ MER : " + tokens[2] + " MAE : " + tokens[3] + " ]";

                    printWriter.println(actualSeq);
                    printWriter.println(predictedSeq);
                    printWriter.println(errorOutput);
                    printWriter.flush();
                }

                String errorOutputString = "[ AVG.MER : " +
                        Precision.round(totalMER / results.size(), 2, BigDecimal.ROUND_HALF_UP) +
                        "  AVG.MAE : " + Precision.round(totalMAE / results.size(), 2, BigDecimal.ROUND_HALF_UP) + " ]";

                printWriter.println(errorOutputString);
                printWriter.flush();

                printWriter.close();

            } catch (IOException e) {
                e.printStackTrace();
                System.exit(1);
            }

        }
    }

}
