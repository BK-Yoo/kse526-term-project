/*        Copyright [BKYoo]
 *
 *        Licensed under the Apache License, Version 2.0 (the "License");
 *        you may not use this file except in compliance with the License.
 *        You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *        Unless required by applicable law or agreed to in writing, software
 *        distributed under the License is distributed on an "AS IS" BASIS,
 *         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *        See the License for the specific language governing permissions and
 *        limitations under the License.
 */

package com.kse.bigdata.main;

import com.kse.bigdata.entity.Sequence;
import com.kse.bigdata.file.SequenceSampler;
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
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 * Created by user on 2014-12-02.
 * KSE526 Term Project
 */
public class Driver {

//    public static class Combiner extends Reducer<NullWritable, Text, NullWritable, Text>{
//
//        private SortedSet<Sequence> neighbors = new TreeSet<>();
//        private final int BUFFER_SIZE = 200;
//
//        @Override
//        protected void reduce(NullWritable ignore, Iterable<Text> values, Context context)
//                throws IOException, InterruptedException {
//
//            for(Text value : values) {
//                Sequence newSeq = new Sequence(value.toString());
//                addSeqToNeighbors(newSeq);
//
//                if(newSeq.getEuclideanDistance() <= neighbors.last().getEuclideanDistance())
//                    context.write(NullWritable.get(), value);
//
//            }
//        }
//
//        private void addSeqToNeighbors(Sequence newSeq){
//            neighbors.add(newSeq);
//
//            if(neighbors.size() > BUFFER_SIZE)
//                neighbors.remove(neighbors.last());
//        }
//    }

    public static class Map extends Mapper<LongWritable, Text, NullWritable, Text> {
        public static final String NUMBER_OF_NEAREST_NEIGHBOR   = "nnn";
        public static final String NORMALIZATION                = "normalize";
        public static final String INPUT_SEQUENCE               = "inputSeq";

        //parameters which user set.
        //get the initial values in the setup method.
        private boolean normalization         = false;
        private int NUMBER_OF_NEIGHBOR        = 100;
        private Sequence userInputSequence;

        private int previousSourceFileId      = -1;

        private final Text                result     = new Text();
        private final LinkedList<Double>  buffer     = new LinkedList<>();
        private final SortedSet<Sequence> neighbors  = new TreeSet<>();

        private final StandardDeviation standardDeviation = new StandardDeviation();
        private final Mean              mean              = new Mean();
        private final EuclideanDistance euclideanDistance = new EuclideanDistance();

        @Override
        public void setup(Context context) throws IOException{
            userInputSequence      = new Sequence(context.getConfiguration().get(INPUT_SEQUENCE, ""));
            NUMBER_OF_NEIGHBOR     = context.getConfiguration().getInt(NUMBER_OF_NEAREST_NEIGHBOR, 100);
            normalization          = context.getConfiguration().getBoolean(NORMALIZATION, false);
        }


        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
            String[] tokens = value.toString().split(",");
            Double windSpeed = Double.valueOf(tokens[0]);
            int currentSourceFileId = Integer.parseInt(tokens[1]);

            if(currentSourceFileId != previousSourceFileId){
                buffer.clear();
                previousSourceFileId = currentSourceFileId;
            }

            buffer.add(windSpeed);

            if(buffer.size() == Sequence.SIZE_OF_SEQUENCE){
                Sequence newSeq = new Sequence(buffer);
                newSeq.setEuclideanDistance(calculateEuclideanDistance(newSeq, userInputSequence));
                addSeqToNeighbors(newSeq);
                buffer.removeFirst();
            }

        }

        private void addSeqToNeighbors(Sequence newSeq){
            neighbors.add(newSeq);

            if(neighbors.size() > NUMBER_OF_NEIGHBOR)
                neighbors.remove(neighbors.last());
        }

        @Override
        public void cleanup(Context context) throws InterruptedException, IOException{
            for(Sequence seq : neighbors) {
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

    public static class Reduce extends Reducer<NullWritable, Text, Text, Text> {
        public static final String MEDIAN                      = "median";
        public static final String NUMBER_OF_NEAREAST_NEIGHBOR = "nnn";
        public static final String INPUT_SEQUENCE              = "inputSeq";

        //parameters which user set.
        //get the initial values in the setup method.
        private int     NUMBER_OF_NEIGHBOR = 0;
        private boolean USE_MEDIAN         = false;

        //Parameters for rounding method.
        private final int    DECIMAL_SCALE   = 3;
        private final int    ROUNDING_METHOD = BigDecimal.ROUND_HALF_UP;
        private final String WHITE_SPACE     = " ";

        // Variables for reducing the cost of creating instance.
        private final Text predictedSeq   = new Text();
        private final Text error          = new Text();

        private Sequence userInputSequence;
        private final SortedSet<Sequence> neighbors = new TreeSet<>();

        private final Mean   mean     = new Mean();
        private final Median median   = new Median();

        @Override
        public void setup(Context context) throws IOException{
            // one for same user input
            NUMBER_OF_NEIGHBOR = context.getConfiguration().getInt(NUMBER_OF_NEAREAST_NEIGHBOR, 100) + 1;
            USE_MEDIAN = context.getConfiguration().getBoolean(MEDIAN, false);
            userInputSequence  = new Sequence(context.getConfiguration().get(INPUT_SEQUENCE, ""));
        }

        @Override
        protected void reduce(NullWritable ignore, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            for(Text value : values) {
                Sequence newSeq = new Sequence(value.toString());
                addSeqToNeighbors(newSeq);
            }
        }

        private void addSeqToNeighbors(Sequence newSeq){
            neighbors.add(newSeq);

            if(neighbors.size() > NUMBER_OF_NEIGHBOR)
                neighbors.remove(neighbors.last());
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            double[] predictedValues = predictSequence();

            StringBuilder stringBuilder =  new StringBuilder();
            for(int index = 0; index < predictedValues.length; index++){
                stringBuilder.append(predictedValues[index]);

                if(index == (predictedValues.length - 1))
                    break;

                stringBuilder.append(Sequence.DELIMITER);
            }

            double MER = calculateMeanErrorRate(predictedValues, userInputSequence.getTail());
            double MAE = calculateMeanAbsoluteError(predictedValues, userInputSequence.getTail());

            predictedSeq.set(stringBuilder.toString());
            error.set(String.valueOf(MER) + WHITE_SPACE + String.valueOf(MAE));
            context.write(predictedSeq, error);
        }

        private double[] predictSequence(){
            double[] predictionResult = new double[Sequence.SIZE_OF_TAIL_SEQ];
            double[][] tails = new double[Sequence.SIZE_OF_TAIL_SEQ][];

            double sumOfWeights = 0.0d;
            double[] tailOfSeq;
            double weight;

            int counter = 0;
            for(Sequence seq : this.neighbors) {
                tailOfSeq = seq.getTail();

                if (USE_MEDIAN) {
                    for (int index = 0; index < Sequence.SIZE_OF_TAIL_SEQ; index++) {
                        if(tails[index] == null)
                            tails[index] = new double[this.neighbors.size()];

                        tails[index][counter] = tailOfSeq[index];
                    }

                    counter++;

                } else {
                    weight = 1.0d;//calculateWeight();
                    sumOfWeights += weight;

                    for (int index = 0; index < Sequence.SIZE_OF_TAIL_SEQ; index++) {
                        predictionResult[index] += weight * tailOfSeq[index];
                    }
                }
            }

            for (int index = 0; index < Sequence.SIZE_OF_TAIL_SEQ; index++) {
                if(USE_MEDIAN) {
                    predictionResult[index] = median.evaluate(tails[index]);

                } else {
                    predictionResult[index] = Precision.round((predictionResult[index] / sumOfWeights),
                            DECIMAL_SCALE, ROUNDING_METHOD);
                }
            }

            return predictionResult;
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

//        private Log  log      = new Log();
//        private double calculateWeight(){
//            double eucDist = seq.getEuclideanDistance();

//            if(eucDist == 0)
//                eucDist = 0.0001;

//            return 1.0d;//log.value(1.0d/eucDist);
//        }

    }

    public static void main(String[] args) throws Exception {

        /**********************************************************************************
         **    Merge the source files into one.                                          **
        /**    Should change the directories of each file before executing the program   **
        ***********************************************************************************/
//        String inputFileDirectory = "/media/bk/드라이브/BigData_Term_Project/Debug";
//        String resultFileDirectory = "/media/bk/드라이브/BigData_Term_Project/debug.csv";
//        File resultFile = new File(resultFileDirectory);
//        if(!resultFile.exists())
//            new SourceFileMerger(inputFileDirectory, resultFileDirectory).mergeFiles();



        /**********************************************************************************
         * Hadoop Operation.
         * Befort Start, Check the Length of Sequence We Want to Predict.
         **********************************************************************************/

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

            /*
             * Mandatory command
             */
            //Extract input path string from command line.
            if (args[index].equals("-in"))
                inputPath = args[index + 1];

            //Extract output path string from command line.
            if (args[index].equals("-out"))
                outputPath = args[index + 1];

            //Extract test data path string from command line.
            if (args[index].equals("-test"))
                testPath = args[index + 1];

            /*
             * Optional command
             */
            //Extract a number of neighbors.
            if (args[index].equals("-nn"))
                conf.setInt(Reduce.NUMBER_OF_NEAREAST_NEIGHBOR, Integer.parseInt(args[index + 1]));

            //Whether job uses normalization or not.
            if(args[index].equals("-norm"))
                conf.setBoolean(Map.NORMALIZATION, true);

            //Extract the number of sample size to test.
            if (args[index].equals("-s"))
                sampleSize = Integer.valueOf(args[index + 1]);

            //Whether job uses mean or median
            //[Default : mean]
            if(args[index].equals("-med"))
                conf.setBoolean(Reduce.MEDIAN, true);
        }

        String outputFileName = "part-r-00000";
        SequenceSampler sampler = new SequenceSampler(testPath, sampleSize);
        LinkedList<Sequence> testSequences = sampler.getRandomSample();

//        Test Sequence
//        String testSeqString = "13.591-13.674-13.778-13.892-13.958-14.049-14.153-14.185-14.169-14.092-13.905-13.702-13.438-13.187-13.0-12.914-12.868-12.766-12.62-12.433-12.279-12.142-12.063-12.025-100";
//        Sequence testSeq = new Sequence(testSeqString);
//        LinkedList<Sequence> testSequences = new LinkedList<>();
//        testSequences.add(testSeq);

        for (Sequence seq : testSequences) {

            /*
             ********************  Hadoop Launch ***********************
             */

            System.out.println(seq.getTailString());

            conf.set(Map.INPUT_SEQUENCE, seq.toString());

            Job job = new Job(conf);
            job.setJarByClass(Driver.class);
            job.setJobName("term-project-driver");

            job.setMapperClass(Map.class);
            job.setMapOutputKeyClass(NullWritable.class);
            job.setMapOutputValueClass(Text.class);

//          Should think another way to implement the combiner class
//          Current Implementation is not helpful to Job.
//          job.setCombinerClass(Combiner.class);

            //Set 1 for number of reduce task for keeping 100 most neighbors in sorted set.
            job.setNumReduceTasks(1);
            job.setReducerClass(Reduce.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

            job.setInputFormatClass(TextInputFormat.class);
            job.setOutputFormatClass(TextOutputFormat.class);

            FileInputFormat.setInputPaths(job, new Path(inputPath));
            FileOutputFormat.setOutputPath(job, new Path(outputPath));

            job.waitForCompletion(true);

            /*
             * if job finishes, get result of the job and store it in results(list).
             */
            try (FileSystem hdfs = FileSystem.get(new Configuration())) {

                BufferedReader fileReader = new BufferedReader(new InputStreamReader(
                        hdfs.open(new Path(outputPath + "/" + outputFileName))));

                String line;
                while ((line = fileReader.readLine()) != null) {
                    results.add(seq.getSeqString() + " " + line);
                }

                fileReader.close();

                hdfs.delete(new Path(outputPath), true);

            } catch (IOException e) {
                e.printStackTrace();
                System.exit(1);
            }
        }

        /*
         * if all jobs finish, store results of jobs to output/result.txt file.
         */
        String finalOutputPath = "output/result.csv";
        try (FileSystem hdfs = FileSystem.get(new Configuration())) {

            Path file = new Path(finalOutputPath);
            if (hdfs.exists(file)) {
                hdfs.delete(file, true);
            }

            OutputStream os = hdfs.create(file);
            PrintWriter printWriter = new PrintWriter(new OutputStreamWriter(os, "UTF-8"));

            //CSV File Header
            printWriter.println("Actual,Predicted,MER,MAE");
            printWriter.flush();

            for (String result : results) {
                String[] tokens = result.split("\\s+");

                printWriter.println(tokens[0] + "," + tokens[1] + "," + tokens[2] +"," + tokens[3]);
                printWriter.flush();
            }

            printWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }


    }

}