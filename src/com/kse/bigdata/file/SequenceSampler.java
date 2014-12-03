package com.kse.bigdata.file;

import com.kse.bigdata.entity.Sequence;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;


/**
 * Created by bk on 14. 12. 3.
 */
public class SequenceSampler {

    private final Path sampleFile;

    private final int INDEX_OF_POWER_GENERATION_INFO = 3;
    private final String DELIMITER = ",";
    private final int TOTAL_SEQUENCE_LENGTH = 157971;
    private final int SAMPLE_SIZE = 1;
    private final String EXCLUDE_LINE_PATTERN = "^(DATE|SITE).*$";

    private LinkedList<Sequence> randomSamples;

    public SequenceSampler(String sampleDirectory){
        sampleFile = new Path(sampleDirectory);
        randomSamples =  new LinkedList<>();
    }

    public LinkedList<Sequence> getRandomSample(){
        try ( FileSystem fs = FileSystem.get(new Configuration());
                BufferedReader fileReader = new BufferedReader(new InputStreamReader(fs.open(sampleFile)))) {

            LinkedList<Double> deque = new LinkedList<>();
            String line = "";
            int[] sampleIndexes = getRandomSampleIndexArray();
            int counter = -1;

            //Read the file line by line
            while((line=fileReader.readLine())!=null){
                counter++;

                if(line.matches(EXCLUDE_LINE_PATTERN))
                    continue;

                deque.add(extractValidInformation(line));

                if (deque.size() == Sequence.SIZE_OF_SEQUENCE) {

                    for(int sampleIndex : sampleIndexes)
                        if(sampleIndex == counter)
                            randomSamples.add(new Sequence(deque));

                    deque.removeFirst();
                }

                if(randomSamples.size() == SAMPLE_SIZE)
                    return randomSamples;

            }
        } catch(IOException e) {
            e.printStackTrace();
        }

        return this.randomSamples;
    }

    private int[] getRandomSampleIndexArray(){
        int[] sampleIndexes = new int[SAMPLE_SIZE * 2];
        RandomGenerator lotto = new JDKRandomGenerator();

        for(int index = 0; index < SAMPLE_SIZE * 2; index ++) {
            sampleIndexes[index] = lotto.nextInt(TOTAL_SEQUENCE_LENGTH);
        }

        return sampleIndexes;
    }


    private double extractValidInformation(String line) throws IOException{
        String powerGenerationData = line.split(DELIMITER)[INDEX_OF_POWER_GENERATION_INFO];
        if(powerGenerationData.startsWith("."))
            powerGenerationData = "0".concat(powerGenerationData);

        return Double.valueOf(powerGenerationData);
    }
}
