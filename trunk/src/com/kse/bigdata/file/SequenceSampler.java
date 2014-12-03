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
 * KSE526 Term Project
 */
public class SequenceSampler {

    private final Path sampleFile;

    private final int INDEX_OF_POWER_GENERATION_INFO = 3;
    private final String DELIMITER = ",";
    private final int TOTAL_SEQUENCE_LENGTH = 157971;
    private final int SAMPLE_SIZE;
    private final String EXCLUDE_LINE_PATTERN = "^(DATE|SITE).*$";

    private LinkedList<Sequence> randomSamples;

    public SequenceSampler(String sampleDirectory, int sampleSize){
        SAMPLE_SIZE = sampleSize;
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
                        if(/*sampleIndex*/60 == counter)
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
        int[] sampleIndexes = new int[SAMPLE_SIZE];
        RandomGenerator lotto = new JDKRandomGenerator();

        for(int index = 0; index < SAMPLE_SIZE; index ++) {

            int randomIndex = 0;
            boolean ok = true;

            while(ok){
                randomIndex = lotto.nextInt(TOTAL_SEQUENCE_LENGTH);

                if(randomIndex + 35 < TOTAL_SEQUENCE_LENGTH)
                    ok = false;
            }

            sampleIndexes[index] = randomIndex;
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
