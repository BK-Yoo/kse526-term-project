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

package com.kse.bigdata.file;

import java.io.*;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by user on 2014-12-02.
 * KSE526 Term Project
 */
public class SourceFileMerger {

    public final String DELIMITER = ",";
    public final String SOURCE_FILE_NAME_PATTERN = "^SITE_.*$";
    public final String EXCLUDE_LINE_PATTERN = "^(DATE|SITE).*$";
    public final int INDEX_OF_POWER_GENERATION_INFO = 3;

    private final File inputFolder;
    private final File outputFile;
    private ArrayList<File> sourceFiles;

    private Pattern sourceFIleNumberPattern = Pattern.compile("\\_(.*?)\\.");

    public SourceFileMerger(String inputDirectory, String outputDirectory){
        this.inputFolder = new File(inputDirectory);
        this.outputFile  = new File(outputDirectory);
        this.sourceFiles = new ArrayList<File>();
        addFilesInDirectory(this.inputFolder);
    }

    public void mergeFiles(){
        try{
            PrintWriter printWriter  = new PrintWriter(outputFile);
            for(File sourceFile : sourceFiles) {
                System.out.println(sourceFile.getName());

                try{
                    BufferedReader fileReader = new BufferedReader(new FileReader(sourceFile));
                    //Read the file line by line
                    String line;
                    while ((line = fileReader.readLine()) != null) {

                        if(line.matches(EXCLUDE_LINE_PATTERN))
                            continue;

                        printWriter.println(extractValidInformation(line, sourceFile.getName()));
                        printWriter.flush();
                    }

                } catch (Exception e) {
                    e.printStackTrace();
                }

            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void addFilesInDirectory(File targetDirectory){
        if(targetDirectory.isDirectory()){
            for(File subDirectory : targetDirectory.listFiles())
                addFilesInDirectory(subDirectory);

        } else if(targetDirectory.isFile()) {
            if(targetDirectory.getName().matches(SOURCE_FILE_NAME_PATTERN))
                this.sourceFiles.add(targetDirectory);
        }
    }

    private String extractValidInformation(String line, String sourceFileName) throws IOException{
        try {
            Matcher matcher = sourceFIleNumberPattern.matcher(sourceFileName);
            if(matcher.find()) {
                String fileNumber = matcher.group(1).replaceAll("0", "");

                String powerGenerationData = line.split(DELIMITER)[INDEX_OF_POWER_GENERATION_INFO];
                if(powerGenerationData.startsWith("."))
                    powerGenerationData = "0".concat(powerGenerationData);

                return powerGenerationData + "," + fileNumber;
            }

            throw new IOException();

        } catch (ArrayIndexOutOfBoundsException e){
            System.out.println(line);
            throw new IOException();
        }
    }

}