package com.kse.bigdata.main;

import com.kse.bigdata.file.SourceFileMerger;

import java.io.File;

/**
 * Created by user on 2014-12-02.
 */
public class Driver {

    public static void main(String[] args){
        /**
         * Merge the source files into one.
         */
        //##################################################################################
        //##    Should change the directories of each file before executing the program   ##
        //##################################################################################
        String inputFileDirectory = "D:\\BigData_Term_Project\\Data";
        String resultFileDirectory = "D:\\BigData_Term_Project\\Merge_Result.csv";
        File resultFile = new File(resultFileDirectory);
        if(!resultFile.exists())
            new SourceFileMerger(inputFileDirectory, resultFileDirectory).mergeFiles();

        /**
         * Hadoop Operation.
         */

    }
}
