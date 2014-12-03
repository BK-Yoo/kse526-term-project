package com.kse.bigdata.main;

import com.kse.bigdata.entity.Sequence;
import com.kse.bigdata.file.SequenceSampler;
import org.apache.commons.math3.analysis.function.Log;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;
import org.apache.hadoop.conf.Configuration;
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

import java.io.IOException;
import java.math.BigDecimal;
import java.util.*;

/**
 * Created by user on 2014-12-02.
 * KSE526 Term Project
 */
public class Driver {

    public static class Map extends Mapper<LongWritable, Text, NullWritable, Text> {
        public static final String NORMALIZATION                = "normalize";
        public static final String INPUT_SEQUENCE               = "inputSeq";
        public static final String EUCLIDEAN_DISTANCE_THRESHOLD = "eucDist";

        private double euclideanDistThreshold = 0.0d;
        private boolean normalization         = false;

        private Sequence userInputSequence;

        private Text result = new Text();

        private LinkedList<Double> temp;
        private LinkedList<Sequence> sequences = new LinkedList<>();
        private HashMap<Integer, LinkedList<Double>> tempSequence = new HashMap<>();

        private StandardDeviation standardDeviation = new StandardDeviation();
        private Mean              mean              = new Mean();
        private EuclideanDistance euclideanDistance = new EuclideanDistance();

        @Override
        public void setup(Context context) throws IOException{
            euclideanDistThreshold = 1.0d * context.getConfiguration().getInt(EUCLIDEAN_DISTANCE_THRESHOLD, 10);
            userInputSequence      = new Sequence(context.getConfiguration().get(INPUT_SEQUENCE, ""));
            normalization = context.getConfiguration().getBoolean(NORMALIZATION, false);
        }


        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
            String[] tokens = value.toString().split(",");
            double powerGeneration = Double.valueOf(tokens[0]);
            int sourceFileId = Integer.valueOf(tokens[1]);

            if(tempSequence.containsKey(sourceFileId)) {
                tempSequence.get(sourceFileId).add(powerGeneration);

            } else {
                tempSequence.put(sourceFileId, new LinkedList<Double>());

            }

            temp = tempSequence.get(sourceFileId);

            if(temp.size() == Sequence.SIZE_OF_SEQUENCE){
                Sequence newSeq = new Sequence(temp);
                newSeq.setEuclideanDistance(calculateEuclideanDistance(newSeq, userInputSequence));

                if(newSeq.getEuclideanDistance() <= euclideanDistThreshold)
                    sequences.add(newSeq);

                temp.removeFirst();
            }

        }

        @Override
        public void cleanup(Context context) throws InterruptedException, IOException{
            for(Sequence seq : sequences) {
                System.out.println("seq:" + seq.getEuclideanDistance());
                result.set(seq.toString());
                context.write(NullWritable.get(), result);
            }
        }

        private double calculateEuclideanDistance(Sequence seqA, Sequence seqB){
            if(normalization) {
                seqA.setNormTail(normalize(seqA.getHead()));
                seqB.setNormTail(normalize(seqB.getHead()));
            }

            return euclideanDistance.compute(seqA.getTail(normalization), seqB.getTail(normalization));
        }

        private double[] normalize(double[] targetSeq){
            if(!normalization)
                return targetSeq;

            double[] normHead = new double[Sequence.SIZE_OF_HEAD_SEQ];
            double avg = mean.evaluate(targetSeq);
            double std = standardDeviation.evaluate(targetSeq);

            for(int index = 0; index < Sequence.SIZE_OF_HEAD_SEQ; index++){
                normHead[index] = (targetSeq[index] - avg) / std;
            }

            return normHead;
        }
    }

    public static class Reduce extends Reducer<NullWritable, Text, Text, Text> {
        public static final String NUMBER_OF_NEAREAST_NEIGHBOR = "nnn";
        public static final String NORMALIZATION               = "normalize";
        public static final String INPUT_SEQUENCE              = "inputSeq";

        private final int DECIMAL_SCALE = 2;
        private final int ROUNDING_METHOD = BigDecimal.ROUND_HALF_UP;

        // Constant for checking the size of sequences.
        protected int NUMBER_OF_NEIGHBOR = 0;

        // Variables for reducing the cost of creating instance.
        private Text tempKey = new Text();
        private Text tempValue = new Text();

        private Sequence userInputSequence;
        private boolean normalization;

        // Container of word for 100 most frequent sequences.
        private SortedSet<Sequence> sequences = new TreeSet<>();

        private Mean mean = new Mean();
        private Log log   = new Log();

        @Override
        public void setup(Context context) throws IOException{
            NUMBER_OF_NEIGHBOR = context.getConfiguration().getInt(NUMBER_OF_NEAREAST_NEIGHBOR, 100);
            userInputSequence  = new Sequence(context.getConfiguration().get(INPUT_SEQUENCE, ""));
            normalization = context.getConfiguration().getBoolean(NORMALIZATION, false);
        }

        @Override
        protected void reduce(NullWritable ignore, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for(Text value : values) {
                Sequence newSeq = new Sequence(value.toString());

                if (sequences.isEmpty()) {
                    sequences.add(newSeq);

                } else if (newSeq.getEuclideanDistance() <= sequences.last().getEuclideanDistance()) {
                    addWordToSortedSet(newSeq);
                }
            }
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException { emitWords(context); }

        private void emitWords(Context context) throws IOException, InterruptedException {
            double[] predictedValues = predictSequence(this.sequences);

            String textResult = "";
            for(int index = 0; index < predictedValues.length; index++){
                textResult += predictedValues[index];

                if(index == (predictedValues.length - 1))
                    break;

                textResult += "-";
            }

            double MER = calculateMeanErrorRate(predictedValues, userInputSequence.getTail(false));
            double MAE = calculateMeanAbsoluteError(predictedValues, userInputSequence.getTail(false));

            tempKey.set(textResult);
            tempValue.set(String.valueOf(MER) + " " + String.valueOf(MAE));
            context.write(tempKey, tempValue);
        }

        private double[] predictSequence(SortedSet<Sequence> sequences){
            double[] predictionResult = new double[Sequence.SIZE_OF_TAIL_SEQ];
            double sumOfWeights = 0.0d;

            double meanOfUserInputSequence = 0.0d;
            double[] tailOfSeq;
            double weight;
            for(Sequence seq : sequences) {

                if (normalization)
                    meanOfUserInputSequence = mean.evaluate(userInputSequence.getHead());

                tailOfSeq = seq.getTail(normalization);
                weight = calculateWeight(true, seq);
                sumOfWeights += weight;

                for (int index = 0; index < Sequence.SIZE_OF_TAIL_SEQ; index++) {
                    predictionResult[index] += weight * tailOfSeq[index];
                }
            }

            for(int index = 0; index < Sequence.SIZE_OF_TAIL_SEQ; index++)
                predictionResult[index] = Precision.round(
                        (meanOfUserInputSequence + predictionResult[index]) / sumOfWeights,
                        DECIMAL_SCALE, ROUNDING_METHOD);

            return predictionResult;
        }

        private void addWordToSortedSet(Sequence newSeq){
            sequences.add(newSeq);

            if(sequences.size() > NUMBER_OF_NEIGHBOR)
                // last element has the smallest frequency among the sequences.
                sequences.remove(sequences.last());
        }

        private double calculateWeight(boolean isNormalized, Sequence seq){
            return isNormalized? log.value(1.0d/seq.getEuclideanDistance()) : 1.0d;
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
//        String inputFileDirectory = "/media/bk/드라이브/BigData_Term_Project/Test";
//        String resultFileDirectory = "/media/bk/드라이브/BigData_Term_Project/Debug_Test.csv";
//        File resultFile = new File(resultFileDirectory);
//        if(!resultFile.exists())
//            new SourceFileMerger(inputFileDirectory, resultFileDirectory).mergeFiles();

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

        for(int index = 0; index < args.length; index++){

            /**
             * Mandatory command
             */
            //Extract input path string from command line.
            if(args[index].equals("-in"))
                inputPath = args[index + 1];

            //Extract output path string from command line.
            if(args[index].equals("-out"))
                outputPath = args[index + 1];

            if(args[index].equals("-test"))
                testPath = args[index + 1];

            /**
             * Optional command
             */
            //Extract the length of target words.
            if(args[index].equals("-dist")) {
                conf.setInt(Map.EUCLIDEAN_DISTANCE_THRESHOLD, Integer.valueOf(args[index + 1]));
            }
            //Number of neighbor
            if(args[index].equals("-nn"))
                conf.setInt(Reduce.NUMBER_OF_NEAREAST_NEIGHBOR, Integer.valueOf(args[index + 1]));

            //Normalization
            if(args[index].equals("-norm"))
                conf.setBoolean(Map.NORMALIZATION, true);
        }

        String outputFileName = "part-r-00000";
        String finalOutputPath = "result";
        ArrayList<String> results = new ArrayList<>();
        SequenceSampler sampler = new SequenceSampler(testPath);
        LinkedList<Sequence> testSequences = sampler.getRandomSample();

        for(Sequence seq : testSequences) {
            System.out.println("Random Sample : " + seq.toString());
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

//            try(FileSystem hdfs = FileSystem.get(new Configuration());) {
//
//                BufferedReader fileReader = new BufferedReader(new InputStreamReader(
//                        hdfs.open(new Path(outputPath + "/" + outputFileName))));
//
//                String line = "";
//                while((line=fileReader.readLine())!=null) {
//                    results.add(line);
//                }
//
//                fileReader.close();
//
//                hdfs.delete(new Path(outputPath), true);
//
//            } catch (IOException e) {
//                e.printStackTrace();
//                System.exit(1);
//            }
        }
//
//        try(FileSystem hdfs = FileSystem.get(new Configuration());) {
//
//            Path file = new Path(finalOutputPath);
//            if(hdfs.exists(file)) { hdfs.delete( file, true);}
//
//            OutputStream os = hdfs.create(file);
//            PrintWriter fileWriter = new PrintWriter(new OutputStreamWriter(os, "UTF-8"));
//
//            for(String result : results){
//                String[] tokens = result.split("\\s+");
//                String outputString = "seq : " + tokens[0] + "MER : " + tokens[1] + "MAE : " + tokens[2];
//                fileWriter.println(outputString);
//                fileWriter.flush();
//            }
//
//            fileWriter.close();
//
//        } catch (IOException e) {
//            e.printStackTrace();
//            System.exit(1);
//        }


    }

}
