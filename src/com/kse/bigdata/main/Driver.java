package com.kse.bigdata.main;

import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
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
import java.util.HashMap;
import java.util.LinkedList;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 * Created by user on 2014-12-02.
 */
public class Driver {

    public static class Sequence implements Comparable<Sequence>{
        private static final int SIZE_OF_SEQUENCE = 36;
        private static final int SIZE_OF_HEAD_SEQ = 6;
        private static final int SIZE_OF_TAIL_SEQ = 30;

        private double euclideanDistance = 100.0d;
        private double[] sequence = new double[SIZE_OF_SEQUENCE];
        private double[] head     = new double[SIZE_OF_HEAD_SEQ];
        private double[] tail     = new double[SIZE_OF_TAIL_SEQ];
        private double[] normTail = null;

        public Sequence(String totalInput) throws IOException{
            if(totalInput.equals(""))
                throw new IOException();

            parseStringToSequence(totalInput);
        }

        public Sequence(LinkedList<Double> inputSequence){
            double value;
            for(int index = 0; index < SIZE_OF_SEQUENCE; index++) {
                value = inputSequence.get(index);
                sequence[index] = value;

                if(index < SIZE_OF_HEAD_SEQ) {
                    head[index] = value;
                } else{
                    tail[index - SIZE_OF_HEAD_SEQ] = value;
                }
            }
        }

        public double getEuclideanDistance() { return this.euclideanDistance; }

        public void setEuclideanDistance(double distance){ this.euclideanDistance = distance; }

        public void setNormTail(double[] normTail){ this.normTail = normTail; }

        public double[] getNormTail() { return this.normTail; }

        /**
         * Parse the sequence data to string.<br>
         * {1,2,3,4,5} will be parsed to "1-2-3-4-5".
         * @return
         */
        @Override
        public String toString(){
            StringBuilder stringBuilder = new StringBuilder();
            for(int index = 0; index < SIZE_OF_SEQUENCE; index++) {
                stringBuilder.append(String.valueOf(sequence[index]));
                if(index == SIZE_OF_SEQUENCE)
                    break;
                stringBuilder.append("-");
            }

            return stringBuilder.toString();
        }

        /**
         * Parse string to sequence.<br>
         * "1-2-3-4-5" will be parsed to {1,2,3,4,5}.
         * @return
         */
        public void parseStringToSequence(String input){
            String[] values = input.split("-");
            double value;
            for(int index = 0; index < SIZE_OF_SEQUENCE; index++) {
                value = Double.valueOf(values[index]);
                sequence[index] = value;

                if(index < SIZE_OF_HEAD_SEQ) {
                    head[index] = value;
                } else {
                    tail[index - SIZE_OF_HEAD_SEQ] = value;
                }
            }
        }


        public double[] getHead(){
            return this.head;
        }

        public double[] getTail(){
            return this.tail;
        }

        @Override
        public int hashCode() {
            return sequence.hashCode();
        }

        @Override
        public boolean equals(Object obj) {
            if(this == obj)
                return true;

            if(obj instanceof Sequence) {

                Sequence seq = ((Sequence) obj);

                for (int index = 0; index < SIZE_OF_SEQUENCE; index++)
                    if (seq.sequence[index] != this.sequence[index])
                        return false;

                return true;

            } else {
                return false;

            }
        }

        @Override
        public int compareTo(Sequence o) {
            return Double.compare(this.euclideanDistance, o.euclideanDistance);

        }

    }

    public static class Map extends Mapper<LongWritable, Text, NullWritable, Text> {
        public static final String NOMALIZATION                 = "normalize";
        public static final String INPUT_SEQUENCE               = "inputSeq";
        public static final String EUCLIDEAN_DISTANCE_THRESHOLD = "eucDist";

        private double euclideanDistThreshold = 0.0d;
        private boolean nomailization         = false;

        private Sequence userInputSequence;
        private HashMap<Integer, LinkedList<Double>> tempSequence = new HashMap<>();

        private LinkedList<Sequence> sequences = new LinkedList<Sequence>();

        private Text result = new Text();

        private StandardDeviation standardDeviation = new StandardDeviation();
        private Mean              mean              = new Mean();
        private EuclideanDistance euclideanDistance = new EuclideanDistance();

        @Override
        public void setup(Context context) throws IOException{
            euclideanDistThreshold = 1.0d * context.getConfiguration().getInt(EUCLIDEAN_DISTANCE_THRESHOLD, 1);
            userInputSequence      = new Sequence(context.getConfiguration().get(INPUT_SEQUENCE, ""));
            nomailization          = context.getConfiguration().getBoolean(NOMALIZATION, false);
        }


        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
            String[] tokens = value.toString().split(",");
            double windSpeed = Double.valueOf(tokens[0]);
            int sourceFileId = Integer.valueOf(tokens[1]);

            if(tempSequence.containsKey(sourceFileId)) {
                tempSequence.get(sourceFileId).add(windSpeed);

            } else {
                tempSequence.put(sourceFileId, new LinkedList<Double>());

            }

            LinkedList<Double> temp = tempSequence.get(sourceFileId);
            if(temp.size() == Sequence.SIZE_OF_SEQUENCE){
                Sequence sequence = new Sequence(temp);
                sequence.setEuclideanDistance(calculateEuclideanDistance(sequence, userInputSequence));
                if(sequence.getEuclideanDistance() <= euclideanDistThreshold)
                    sequences.add(sequence);
                temp.removeFirst();
            }

        }

        @Override
        public void cleanup(Context context) throws InterruptedException, IOException{
            for(Sequence seq : sequences) {
                result.set(seq.toString());
                context.write(NullWritable.get(), result);
            }
        }

        private double calculateEuclideanDistance(Sequence seqA, Sequence seqB){
            if(seqA.getNormTail() == null)
                seqA.setNormTail(normalize(seqA.getHead()));
            if(seqB.getNormTail() == null)
                seqB.setNormTail(normalize(seqB.getHead()));

            return euclideanDistance.compute(seqA.getNormTail(), seqB.getNormTail());
        }

        private double[] normalize(double[] targetSeq){
            if(!nomailization)
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

    public static class Combiner extends Reduce{
        private SortedSet<Sequence> sequences = new TreeSet<>();

        @Override
        protected void reduce(NullWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            for(Text value : values) {
                Sequence newSeq = new Sequence(value.toString());

                if (sequences.isEmpty()) {
                    sequences.add(newSeq);

                } else if (newSeq.getEuclideanDistance() <= sequences.last().getEuclideanDistance()) {
                    addWordToSortedSet(newSeq);
                    context.write(NullWritable.get(), value);
                }
            }

        }

        private void addWordToSortedSet(Sequence newSequence){
            sequences.add(newSequence);

            if(sequences.size() > NUMBER_OF_NEIGHBOR)
                // last element has the smallest frequency among the sequences.
                sequences.remove(sequences.last());
        }
    }

    public static class Reduce extends Reducer<NullWritable, Text, NullWritable, Text> {
        public static final String NUMBER_OF_NEAREAST_NEIGHBOR = "nnn";
        public static final String NORMALIZATION               = "normalize";
        public static final String INPUT_SEQUENCE              = "inputSeq";



        // Constant for checking the size of sequences.
        protected int NUMBER_OF_NEIGHBOR = 0;

        // Variables for reducing the cost of creating instance.
        private Text tempKey = new Text();

        private Sequence userInputSequence;
        private boolean normalization;

        // Container of word for 100 most frequent sequences.
        private SortedSet<Sequence> sequences = new TreeSet<Sequence>();

        private Mean mean = new Mean();

        @Override
        public void setup(Context context) throws IOException{
            NUMBER_OF_NEIGHBOR = context.getConfiguration().getInt(NUMBER_OF_NEAREAST_NEIGHBOR, 100);
            userInputSequence  = new Sequence(context.getConfiguration().get(INPUT_SEQUENCE, ""));
            normalization = context.getConfiguration().getBoolean(NORMALIZATION, false);
        }

        @Override
        protected void reduce(NullWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            for(Text value : values) {
                Sequence newSeq = new Sequence(value.toString());

                if (sequences.isEmpty()) {
                    sequences.add(newSeq);

                } else if (newSeq.getEuclideanDistance() >= sequences.last().getEuclideanDistance()) {
                    addWordToSortedSet(newSeq);
                }
            }
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            emitWords(context);
        }

        private void emitWords(Context context) throws IOException, InterruptedException {
            double[] result = predictSequence(this.sequences);
            tempKey.set(result.toString());
            context.write(NullWritable.get(), tempKey);
        }

        private double[] predictSequence(SortedSet<Sequence> sequences){
            double[] predictionResult = new double[Sequence.SIZE_OF_TAIL_SEQ];
            double sumOfWeights = 0.0d;

            double meanOfUserInputSequence = 0.0d;
            double[] tailOfSeq;
            double weight;
            for(Sequence seq : sequences){

                if(normalization) {
                    meanOfUserInputSequence = mean.evaluate(userInputSequence.getHead());
                    tailOfSeq = seq.getNormTail();
                    weight = seq.getEuclideanDistance();

                } else {
                    tailOfSeq   = seq.getTail();
                    weight      = 1.0d;
                }

                sumOfWeights += weight;

                for(int index = Sequence.SIZE_OF_HEAD_SEQ; index < Sequence.SIZE_OF_SEQUENCE; index++) {
                    predictionResult[index] +=  weight * tailOfSeq[index];
                }
            }

            for(int index = Sequence.SIZE_OF_HEAD_SEQ; index < Sequence.SIZE_OF_SEQUENCE; index++)
                predictionResult[index] += meanOfUserInputSequence + predictionResult[index] / sumOfWeights;

            return predictionResult;
        }

        private void addWordToSortedSet(Sequence newSeq){
            sequences.add(newSeq);

            if(sequences.size() > NUMBER_OF_NEIGHBOR)
                // last element has the smallest frequency among the sequences.
                sequences.remove(sequences.last());
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
//        conf.setBoolean("mapred.compress.map.output", true);
//        conf.set("mapred.map.output.compression.codec", "org.apache.hadoop.io.compress.SnappyCodec");

        //Enable Profiling
        //conf.setBoolean("mapred.task.profile", true);

        String testSeq = "0.62-0.92-1.63-2-1.92-2.38-3.05-3.67-3.89-4.5-" +
                         "5.35-6.01-6.18-7.15-8.79-9.64-10-8.41-9.39-8.18-" +
                         "8.18-9.97-10.26-10.26-11.36-11.55-12.05-12.35-13.24-13.54-"+
                         "13.34-12.95-12.75-11.95-10.75-11.85";

        Path inputPath = null;
        Path outputPath = null;
        for(int index = 0; index < args.length; index++){

            /**
             * Mandatory command
             */
            //Extract input path string from command line.
            if(args[index].equals("-in"))
                inputPath = new Path(args[index + 1]);

            //Extract output path string from command line.
            if(args[index].equals("-out"))
                outputPath = new Path(args[index + 1]);

            //Input sequence
            if(args[index].equals("-seq"))
                conf.set(Map.INPUT_SEQUENCE, testSeq);//args[index + 1]);

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
                conf.setBoolean(Map.NOMALIZATION, true);
        }

        Job job = new Job(conf);
        job.setJarByClass(Driver.class);
        job.setJobName("term-project-driver");

        job.setMapperClass(Map.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        job.setCombinerClass(Combiner.class);

        //Set 1 for number of reduce task for keeping 100 most sequences in sorted set.
        job.setReducerClass(Reduce.class);
        job.setNumReduceTasks(1);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.setInputPaths(job, inputPath);
        FileOutputFormat.setOutputPath(job, outputPath);

        try {
            job.waitForCompletion(true);

        } catch (IOException e){
            e.printStackTrace();
            System.exit(1);
        }
    }

}
