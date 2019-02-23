package com.company;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;

public class NaiveBayes {

    int totalTrngExmpl = 0;
    int sizeOfVocab = 0;
    String path;
    //categoryNames Stores the category names of the newsgroups in order
    //Remember to subtract from the id to get the corresponding category number
    ArrayList<String> categoryNames = new ArrayList<>();
    ArrayList<Triple> AllDocs = new ArrayList<>();//ArrayList that stores the Triples created below
    ArrayList<Triple> test_AllDocs = new ArrayList<>();//ArrayList that stores the Triples created below
    //AllDocsLabels Stores which label is assigned to each document
    //i_th index of ArrayList mapped to j means that the i_th doc was labelled as j
    ArrayList<Integer> getCategoryNumberForThisDoc = new ArrayList<>();
    ArrayList<Integer> test_getCategoryNumberForThisDoc = new ArrayList<>();
    ArrayList<Double> priors;//Stores the prior probabilities
    ArrayList<Integer> numDocsInEachCategory;
    ArrayList<Integer> totalNumOfWordsInEachCategory;
    ArrayList<String> reverseDictionary = new ArrayList<>();//Stores the reverse mapping from numbers to strings
    ArrayList<ArrayList<Integer>> DocsInEachCategory;
    //dictionaryWithCounts - Every bucket in outer arrays List corresponds to a word.
    //That points to a list that contains all the categories from 1 to 20.
    // Every entry in that list is the number of times the word appears in that category
    ArrayList<ArrayList<Integer>> dictionaryWithCounts;
    ArrayList<ArrayList<Double>> MLEValues;//Stores the MLE Probabilities
    ArrayList<ArrayList<Double>> BEValues;//Stores the Bayesian Estimate Probabilities
    ArrayList<ArrayList<Integer>> BE_train_confusionMatrix;
    ArrayList<ArrayList<Integer>> BE_test_confusionMatrix;
    ArrayList<ArrayList<Integer>> MLE_train_confusionMatrix;
    ArrayList<ArrayList<Integer>> MLE_test_confusionMatrix;


    public static void main(String[] args) {
        NaiveBayes myBayes = new NaiveBayes();
        String path = "C:\\Semester 4\\COMS 573\\Lab1\\20newsgroups\\";
        myBayes.readCSVs(path);
    }


    public void readCSVs(String path1) {

        String line = "";
        String cvsSplitBy = ",";
        this.path = path1;

        //Reading and storing the contents of map.csv
        String mapFilePath = path + "map.csv";

        categoryNames.add(0, "");

        try (BufferedReader br = new BufferedReader(new FileReader(mapFilePath))) {
            while ((line = br.readLine()) != null) {

                // use comma as separator
                String[] map = line.split(cvsSplitBy);
//                System.out.println("Category Number: " + map[0] + " , Category name: " + map[1] + "]");
                categoryNames.add(map[1]);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }


        //Read the contents of the train_label.csv
        String trainLabelPath = path + "train_label.csv";
        numDocsInEachCategory = new ArrayList<>(Collections.nCopies(categoryNames.size(), 0));

        DocsInEachCategory = new ArrayList<>();
        for (int i = 0; i < categoryNames.size(); i++) {
            DocsInEachCategory.add(new ArrayList<>());
        }

        getCategoryNumberForThisDoc.add(0);//to offset the fact that the doc numbering starts from 1

        int docNumber = 1;
        try (BufferedReader br = new BufferedReader(new FileReader(trainLabelPath))) {
            while ((line = br.readLine()) != null) {

                // use comma as separator
                String[] categoryNumber = line.split(cvsSplitBy);
//                System.out.println("Mapped to category: " + categoryNumber[0]);
                int categoryNum = Integer.parseInt(categoryNumber[0]);
                //i_th index of getCategoryNumberForThisDoc mapped to j means that the i_th doc was labelled as j
                getCategoryNumberForThisDoc.add(categoryNum);

                Integer numDocsInThisCategory = 0;
                numDocsInThisCategory = numDocsInEachCategory.get(categoryNum);
                numDocsInEachCategory.set(categoryNum, ++numDocsInThisCategory);

                DocsInEachCategory.get(categoryNum).add(docNumber);

                totalTrngExmpl++;
                docNumber++;
            }

        } catch (IOException e) {
            e.printStackTrace();
        }


        //train data.csv and test data.csv are formatted "docIdx, wordIdx, count", where
        //docIdx is the document id,
        //wordIdx represents the word id (in correspondence to vocabulary.txt) and
        //count is the frequency of the word in the document.


        //Read the contents of the train_data.csv
        String trainDataPath = path + "train_data.csv";
        AllDocs.add(new Triple(0, 0));//Dummy data becuase the numbering starts from 1

        try (BufferedReader br = new BufferedReader(new FileReader(trainDataPath))) {

            int lastSeenDoc = 0;
            while ((line = br.readLine()) != null) {
                // use comma as separator
                String[] trainData = line.split(cvsSplitBy);
//                System.out.println("DocNumber: " + trainData[0] + " , WordId: " + trainData[1] + " , Count: " + trainData[2] + "]");
                int docNum = Integer.parseInt(trainData[0]);
                int wordID = Integer.parseInt(trainData[1]);
                int countOfWordId = Integer.parseInt(trainData[2]);

                if (docNum > lastSeenDoc) {
                    //This doc is seen for the first time, so we need to create a new Triple
                    int categoryOfThisDoc = getCategoryNumberForThisDoc.get(docNum);
                    Triple myTriple = new Triple(docNum, categoryOfThisDoc);
                    myTriple.addWordToDoc(new Word(wordID, countOfWordId));
                    AllDocs.add(myTriple);
                    lastSeenDoc = docNum;
                } else {
                    //else, the doc already exists, all we have to do is
                    // update the value for the wordID and the count of the wordID
                    AllDocs.get(docNum).addWordToDoc(new Word(wordID, countOfWordId));
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }


        //Read the contents of the vocabulary.txt
        String vocabularyPath = path + "vocabulary.txt";
        reverseDictionary.add("");//Because offset...
        int wordIndex = 1;

        try (BufferedReader br = new BufferedReader(new FileReader(vocabularyPath))) {
            StringBuilder sb = new StringBuilder();
            line = br.readLine();

            while (line != null) {
//                sb.append(line);
//                sb.append(System.lineSeparator());
                line = br.readLine();
                if (line != null) {
                    reverseDictionary.add(line);
                    wordIndex++;
                    sizeOfVocab++;
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        calculatePriorProbabilities();
    }

    //This method calculates the prior probabilities of the class w_j
    void calculatePriorProbabilities() {

//        //Calculate the total number of words that occur in each category
//        for(int i = 0; i < AllDocs.size(); i++){
//            Triple myTriple = AllDocs.get(i);
//            Integer totalOccOfDoc = docs_j.get(myTriple.docNum);
//            totalOccOfDoc++;
//            docs_j.set(myTriple.docNum, totalOccOfDoc);
//        }

        priors = new ArrayList<>(Collections.nCopies(categoryNames.size(), 0.0));
        for (int i = 1; i < numDocsInEachCategory.size(); i++) {
            double prob = numDocsInEachCategory.get(i) / (double) totalTrngExmpl;
            priors.set(i, prob);
            System.out.println("Category " + i + " Probability: " + prob);
        }

        calculateN();
    }

    void calculateN() {

        boolean tshoot = true;

        totalNumOfWordsInEachCategory = new ArrayList<>(Collections.nCopies(categoryNames.size(), 0));

        /* Algorithm
         * for each category c
         *   for each doc d in category c
         *       for each word w in doc d
         *           count += w.count
         *
         * */

        for (int i = 1; i < totalNumOfWordsInEachCategory.size(); i++) {
            int totalNumOfWordsInThisCategory = 0;
            for (int j = 0; j < DocsInEachCategory.get(i).size(); j++) {
                int nextDocInThisCategory = DocsInEachCategory.get(i).get(j);
                ArrayList<Word> listOfWords = AllDocs.get(nextDocInThisCategory).getWordsWithCounts();
                for (int k = 0; k < listOfWords.size(); k++) {
                    totalNumOfWordsInThisCategory += listOfWords.get(k).getCount();
                }
//                System.out.println("Completed Doc: " + nextDocInThisCategory);
            }
            totalNumOfWordsInEachCategory.set(i, totalNumOfWordsInThisCategory);
        }


//        for (int i = 0; i < AllDocs.size(); i++) {
//            int count = AllDocs.get(i).count;
//            int docNum = AllDocs.get(i).docNum;
//            int prevCount = totalNumOfWordsInEachCategory.get(getCategoryNumberForThisDoc.get(docNum));
//            totalNumOfWordsInEachCategory.set(getCategoryNumberForThisDoc.get(docNum), prevCount + count);
//        }

        calculateNK();
    }

    void calculateNK() {

        boolean tshoot = true;

        dictionaryWithCounts = new ArrayList<>();
        dictionaryWithCounts.add(new ArrayList<>(Collections.nCopies(categoryNames.size(), 0)));//Because offset..
        for (int i = 1; i < reverseDictionary.size(); i++) {
            dictionaryWithCounts.add(new ArrayList<>(Collections.nCopies(categoryNames.size(), 0)));
        }


        for (int i = 0; i < AllDocs.size(); i++) {
            Triple currentTriple = AllDocs.get(i);
            ArrayList<Word> listOfWords = currentTriple.getWordsWithCounts();
            int categoryOfCurrDoc = currentTriple.getCategory();
            for (int j = 0; j < listOfWords.size(); j++) {
                int currWord = listOfWords.get(j).getWordID();
                int currCount = listOfWords.get(j).getCount();
                //Adding the currCount ot the previous count stored in dictionaryWithCounts
                int prevCount = dictionaryWithCounts.get(currWord).get(categoryOfCurrDoc);
                int newCount = prevCount + currCount;
                dictionaryWithCounts.get(currWord).set(categoryOfCurrDoc, newCount);

            }
//            if ((i % 100) == 0) System.out.println("Completed " + i + " docs");
        }

        if (tshoot) System.out.println("DOnE?");
        System.out.println();

        calculateAllProbs();
    }

    void calculateAllProbs() {

        boolean tshoot = false;

        MLEValues = new ArrayList<>();
        MLEValues.add(new ArrayList<>(Collections.nCopies(categoryNames.size(), 0.0)));//Because offset
        for (int i = 1; i < reverseDictionary.size(); i++) {
            MLEValues.add(new ArrayList<>(Collections.nCopies(categoryNames.size(), 0.0)));
        }

        BEValues = new ArrayList<>();
        BEValues.add(new ArrayList<>(Collections.nCopies(categoryNames.size(), 0.0)));//Because offset
        for (int i = 1; i < reverseDictionary.size(); i++) {
            BEValues.add(new ArrayList<>(Collections.nCopies(categoryNames.size(), 0.0)));
        }

        for (int i = 1; i < MLEValues.size(); i++) {
            for (int j = 1; j < MLEValues.get(i).size(); j++) {
                int n_k = dictionaryWithCounts.get(i).get(j);
                int n = totalNumOfWordsInEachCategory.get(j);
                double prob_MLE = n_k / (double) n;
                double prob_BE = (n_k + 1) / (double) (n + sizeOfVocab);

                MLEValues.get(i).set(j, prob_MLE);
                BEValues.get(i).set(j, prob_BE);

            }
            if (tshoot) System.out.println("Completed word: " + i);
        }

        if(tshoot) printProbabilitiesTOFile(MLEValues, "MLEValues");
        if(tshoot) printProbabilitiesTOFile(BEValues, "BEValues");

        BE_checkingResults();
        MLE_checkingResults();
    }

    void BE_checkingResults() {

        boolean tshoot = true;

        System.out.println("********* TRAINING DATA | BAYES ESTIMATE *********");

        if (tshoot) System.out.println("Checking BE Method --- Training Data");
        int accuracy = 0;

        BE_train_confusionMatrix = new ArrayList<>();
        for (int i = 0; i < categoryNames.size(); i++) {
            BE_train_confusionMatrix.add(new ArrayList<>(Collections.nCopies(categoryNames.size(), 0)));
        }

        //Formatting the output correctly
        for (int i = 0; i < BE_train_confusionMatrix.size(); i++) {
            for (int j = 0; j < BE_train_confusionMatrix.get(i).size(); j++) {
                if (i == 0) {
                    BE_train_confusionMatrix.get(i).set(j, j);
                }
                if (j == 0) {
                    BE_train_confusionMatrix.get(i).set(j, i);
                }
            }
        }

        for (int i = 1; i < AllDocs.size(); i++) {
            Triple myTriple = AllDocs.get(i);
            int predictedCategory = BE_classifyDoc(myTriple);
            int trueCategory = myTriple.getCategory();
            if (predictedCategory == trueCategory) accuracy++;
            int prevCount = BE_train_confusionMatrix.get(predictedCategory).get(trueCategory);
            BE_train_confusionMatrix.get(predictedCategory).set(trueCategory, ++prevCount);
        }

        System.out.println("Final count of Accuracy for the Bayesian Estimate on Training Set: " + accuracy);
        System.out.println("Final Accuracy for the Bayesian Estimate on Training Set: " + accuracy / (double) AllDocs.size());
        System.out.println("Printing Confusion Matrix for Bayesian Estimate on Training Set:");
        for (int i = 0; i < BE_train_confusionMatrix.size(); i++) {

            for (int j = 0; j < BE_train_confusionMatrix.get(i).size(); j++) {

                System.out.printf("%4d", BE_train_confusionMatrix.get(i).get(j));
//                System.out.print(BE_train_confusionMatrix.get(i).get(j) + " ");

            }
            System.out.println();
        }

        System.out.println("Printing Class Accuracy values for Bayesian Estimate on Training Set:");
        printClassAccuracy(BE_train_confusionMatrix);

        System.out.println();
        System.out.println("********* TESTING DATA | BAYES ESTIMATE *********");
        //Reading files for Testing Data Now....
        readCSVsForTesting(path);
        if (tshoot) System.out.println("Checking BE Method --- Testing Data");
        accuracy = 0;

        BE_test_confusionMatrix = new ArrayList<>();
        for (int i = 0; i < categoryNames.size(); i++) {
            BE_test_confusionMatrix.add(new ArrayList<>(Collections.nCopies(categoryNames.size(), 0)));
        }
        //Formatting the output correctly
        for (int i = 0; i < BE_test_confusionMatrix.size(); i++) {
            for (int j = 0; j < BE_test_confusionMatrix.get(i).size(); j++) {
                if (i == 0) {
                    BE_test_confusionMatrix.get(i).set(j, j);
                }
                if (j == 0) {
                    BE_test_confusionMatrix.get(i).set(j, i);
                }
            }
        }

        for (int i = 1; i < test_AllDocs.size(); i++) {
            Triple myTriple = test_AllDocs.get(i);
            int predictedCategory = BE_classifyDoc(myTriple);
            int trueCategory = myTriple.getCategory();
            if (predictedCategory == trueCategory) accuracy++;
            int prevCount = BE_test_confusionMatrix.get(predictedCategory).get(trueCategory);
            BE_test_confusionMatrix.get(predictedCategory).set(trueCategory, ++prevCount);
        }

        System.out.println("Final count of Accuracy for the Bayesian Estimate on Testing Set: " + accuracy);
        System.out.println("Final Accuracy for the Bayesian Estimate on Testing Set: " + accuracy / (double) test_AllDocs.size());
        System.out.println("Printing Confusion Matrix for Bayesian Estimate on Training Set:");

        for (int i = 0; i < BE_test_confusionMatrix.size(); i++) {
            for (int j = 0; j < BE_test_confusionMatrix.get(i).size(); j++) {
                System.out.printf("%4d", BE_test_confusionMatrix.get(i).get(j));
//                System.out.print(BE_test_confusionMatrix.get(i).get(j) + " ");

            }
            System.out.println();
        }

        printClassAccuracy(BE_test_confusionMatrix);
    }

    int BE_classifyDoc(Triple myTriple) {

        int predictedCategory = 1;
        ArrayList<Double> w_NB = new ArrayList<>(Collections.nCopies(categoryNames.size(), 0.0));

        for (int i = 1; i < categoryNames.size(); i++) {

            double partialSum = 0.0;
            double posterior = 0.0;

            double prior = priors.get(i);

            ArrayList<Word> listOfWords = myTriple.getWordsWithCounts();
            for (int j = 0; j < listOfWords.size(); j++) {
                int wordId = listOfWords.get(j).getWordID();
                if (wordId > sizeOfVocab) continue;
                double likelihood = BEValues.get(listOfWords.get(j).getWordID()).get(i);
                likelihood = Math.log(likelihood);
                likelihood = listOfWords.get(j).getCount() * likelihood;
                partialSum += likelihood;
            }
            posterior = Math.log(prior) + partialSum;
            if (posterior == 0 || posterior > 1) System.out.println("LOOOOOOOOOOOOOOOOOOL!!!!!!!!!!!!!!");
            w_NB.set(i, posterior);
        }

        for (int i = 2; i < w_NB.size(); i++) {

            if (w_NB.get(i) > w_NB.get(predictedCategory)) {
                predictedCategory = i;
            }
        }

        return predictedCategory;
    }

    void MLE_checkingResults() {

        boolean tshoot = true;

        System.out.println("********* TRAINING DATA | MAX LIKELIHOOD ESTIMATE *********");

        if (tshoot) System.out.println("Checking Max Likelihood Method --- Training Data");
        int accuracy = 0;

        MLE_train_confusionMatrix = new ArrayList<>();
        for (int i = 0; i < categoryNames.size(); i++) {
            MLE_train_confusionMatrix.add(new ArrayList<>(Collections.nCopies(categoryNames.size(), 0)));
        }

        //Formatting the output correctly
        for (int i = 0; i < MLE_train_confusionMatrix.size(); i++) {
            for (int j = 0; j < MLE_train_confusionMatrix.get(i).size(); j++) {
                if (i == 0) {
                    MLE_train_confusionMatrix.get(i).set(j, j);
                }
                if (j == 0) {
                    MLE_train_confusionMatrix.get(i).set(j, i);
                }
            }
        }

        for (int i = 1; i < AllDocs.size(); i++) {
            Triple myTriple = AllDocs.get(i);
            int predictedCategory = MLE_classifyDoc(myTriple);
            int trueCategory = myTriple.getCategory();
            if (predictedCategory == trueCategory) accuracy++;
            int prevCount = MLE_train_confusionMatrix.get(predictedCategory).get(trueCategory);
            MLE_train_confusionMatrix.get(predictedCategory).set(trueCategory, ++prevCount);
        }

        System.out.println("Final count of Accuracy for the Max Likelihood Estimate on Training Set: " + accuracy);
        System.out.println("Final Accuracy for the Max Likelihood Estimate on Training Set: " + accuracy / (double) AllDocs.size());
        System.out.println("Printing Confusion Matrix for Max Likelihood Estimate on Training Set:");
        for (int i = 0; i < MLE_train_confusionMatrix.size(); i++) {

            for (int j = 0; j < MLE_train_confusionMatrix.get(i).size(); j++) {

                System.out.printf("%4d", MLE_train_confusionMatrix.get(i).get(j));
//                System.out.print(BE_train_confusionMatrix.get(i).get(j) + " ");
            }
            System.out.println();
        }

        printClassAccuracy(MLE_train_confusionMatrix);


        System.out.println("********* TESTING DATA | MAX LIKELIHOOD ESTIMATE*********");
        //Reading files for Testing Data Now....
        readCSVsForTesting(path);
        if (tshoot) System.out.println("Checking Max Likelihood Method --- Testing Data");
        accuracy = 0;

        MLE_test_confusionMatrix = new ArrayList<>();
        for (int i = 0; i < categoryNames.size(); i++) {
            MLE_test_confusionMatrix.add(new ArrayList<>(Collections.nCopies(categoryNames.size(), 0)));
        }
        //Formatting the output correctly
        for (int i = 0; i < MLE_test_confusionMatrix.size(); i++) {
            for (int j = 0; j < MLE_test_confusionMatrix.get(i).size(); j++) {
                if (i == 0) {
                    MLE_test_confusionMatrix.get(i).set(j, j);
                }
                if (j == 0) {
                    MLE_test_confusionMatrix.get(i).set(j, i);
                }
            }
        }

        for (int i = 1; i < test_AllDocs.size(); i++) {
            Triple myTriple = test_AllDocs.get(i);
            int predictedCategory = MLE_classifyDoc(myTriple);
            int trueCategory = myTriple.getCategory();
            if (predictedCategory == trueCategory) accuracy++;
            int prevCount = MLE_test_confusionMatrix.get(predictedCategory).get(trueCategory);
            MLE_test_confusionMatrix.get(predictedCategory).set(trueCategory, ++prevCount);
//            System.out.println("What the faaaackk");
        }

        System.out.println("Final count of Accuracy for the Max Likelihood Estimate on Testing Set: " + accuracy);
        System.out.println("Final Accuracy for the Max Likelihood Estimate on Testing Set: " + accuracy / (double) test_AllDocs.size());
        System.out.println("Printing Confusion Matrix for Max Likelihood Estimate on Training Set:");

        for (int i = 0; i < MLE_test_confusionMatrix.size(); i++) {
            for (int j = 0; j < MLE_test_confusionMatrix.get(i).size(); j++) {
                System.out.printf("%4d", MLE_test_confusionMatrix.get(i).get(j));

            }
            System.out.println();
        }

        printClassAccuracy(MLE_test_confusionMatrix);
    }

    int MLE_classifyDoc(Triple myTriple) {

        int predictedCategory = 1;
        ArrayList<Double> w_NB = new ArrayList<>(Collections.nCopies(categoryNames.size(), 0.0));

        for (int i = 1; i < categoryNames.size(); i++) {

            double partialSum = 0.0;
            double posterior = 0.0;

            double prior = priors.get(i);

            ArrayList<Word> listOfWords = myTriple.getWordsWithCounts();
            for (int j = 0; j < listOfWords.size(); j++) {
                int wordId = listOfWords.get(j).getWordID();
                if (wordId > sizeOfVocab) continue;
                double likelihood = MLEValues.get(wordId).get(i);
                if (likelihood == 0) {
                    likelihood = Double.MIN_VALUE;
                }
                likelihood = Math.log(likelihood);
                int count = listOfWords.get(j).getCount();
                likelihood = count * likelihood;
                partialSum += likelihood;
            }
            posterior = Math.log(prior) + partialSum;
            if (posterior == 0 || posterior > 1) System.out.println("LOOOOOOOOOOOOOOOOOOL!!!!!!!!!!!!!!");
            w_NB.set(i, posterior);
        }

        for (int i = 2; i < w_NB.size(); i++) {

            if (w_NB.get(i) > w_NB.get(predictedCategory)) {
                predictedCategory = i;
            }
        }

        return predictedCategory;
    }


    void readCSVsForTesting(String path) {

        String line = "";
        String cvsSplitBy = ",";

        //Read the contents of the test_label.csv
        String testLabelPath = path + "test_label.csv";


        test_getCategoryNumberForThisDoc.add(0);//to offset the fact that the doc numbering starts from 1

        try (BufferedReader br = new BufferedReader(new FileReader(testLabelPath))) {
            while ((line = br.readLine()) != null) {
                // use comma as separator
                String[] categoryNumber = line.split(cvsSplitBy);
//                System.out.println("Mapped to category: " + categoryNumber[0]);
                int categoryNum = Integer.parseInt(categoryNumber[0]);
                //i_th index of getCategoryNumberForThisDoc mapped to j means that the i_th doc was labelled as j
                test_getCategoryNumberForThisDoc.add(categoryNum);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }


        //train data.csv and test data.csv are formatted "docIdx, wordIdx, count", where
        //docIdx is the document id,
        //wordIdx represents the word id (in correspondence to vocabulary.txt) and
        //count is the frequency of the word in the document.


        //Read the contents of the test_data.csv
        String testDataPath = path + "test_data.csv";
        test_AllDocs.add(new Triple(0, 0));//Dummy data becuase the numbering starts from 1

        try (BufferedReader br = new BufferedReader(new FileReader(testDataPath))) {

            int lastSeenDoc = 0;
            while ((line = br.readLine()) != null) {
                // use comma as separator
                String[] trainData = line.split(cvsSplitBy);
//                System.out.println("DocNumber: " + trainData[0] + " , WordId: " + trainData[1] + " , Count: " + trainData[2] + "]");
                int docNum = Integer.parseInt(trainData[0]);
                int wordID = Integer.parseInt(trainData[1]);
                int countOfWordId = Integer.parseInt(trainData[2]);

                if (docNum > lastSeenDoc) {
                    //This doc is seen for the first time, so we need to create a new Triple
                    int categoryOfThisDoc = test_getCategoryNumberForThisDoc.get(docNum);
                    Triple myTriple = new Triple(docNum, categoryOfThisDoc);
                    myTriple.addWordToDoc(new Word(wordID, countOfWordId));
                    test_AllDocs.add(myTriple);
                    lastSeenDoc = docNum;
                } else {
                    //else, the doc already exists, all we have to do is
                    // update the value for the wordID and the count of the wordID
                    test_AllDocs.get(docNum).addWordToDoc(new Word(wordID, countOfWordId));
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    void printClassAccuracy(ArrayList<ArrayList<Integer>> myMatrix) {

        boolean tshoot = false;
        //Initialise a 2-D ArrayList in 1 Line? Hold my beer...well not exactly..my bad
        ArrayList<ArrayList<Integer>> MyMatrixTransposed;
        MyMatrixTransposed = new ArrayList<>();
        for (int i = 0; i < myMatrix.size(); i++) {
            MyMatrixTransposed.add(new ArrayList<>(Collections.nCopies(categoryNames.size(), 0)));
        }

        for (int i = 0; i < myMatrix.size(); i++) {
            for (int j = 0; j < myMatrix.get(i).size(); j++) {
                int element = myMatrix.get(i).get(j);
                MyMatrixTransposed.get(j).set(i, element);
            }
        }

        //Printing out the MyMatrixTransposed
        if (tshoot) {
            System.out.println("Printing out the MyMatrixTransposed");
            for (int i = 0; i < MyMatrixTransposed.size(); i++) {
                for (int j = 0; j < MyMatrixTransposed.get(i).size(); j++) {
                    System.out.printf("%4d", MyMatrixTransposed.get(i).get(j));
                }
                System.out.println();
            }
        }

        //Printing out the class accuracy values
        System.out.println();
        int rowTotal = 0;
        for (int i = 1; i < MyMatrixTransposed.size(); i++) {
            rowTotal = 0;
            for (int j = 1; j < MyMatrixTransposed.get(i).size(); j++) {
                rowTotal += MyMatrixTransposed.get(i).get(j);
            }
            System.out.println("Class Accuracy for Category " + i + ": " + MyMatrixTransposed.get(i).get(i) / (double) rowTotal);
        }
        System.out.println();
        System.out.println("******************************************");
        System.out.println();

    }

    void printProbabilitiesTOFile(ArrayList<ArrayList<Double>> myMatrix, String fileName) {


        // Store current System.out before assigning a new value
        PrintStream console = System.out;
        // Creating a File object that represents the disk file.
        try {
            PrintStream o = new PrintStream(new File(fileName));
            // Assign o to output stream
            System.setOut(o);
            System.out.println("This will be written to the text file");
            for (int i = 0; i < myMatrix.size(); i++) {
                for (int j = 0; j < myMatrix.get(i).size(); j++) {
                    System.out.printf("%6f", myMatrix.get(i).get(j));
                }
                System.out.println();
            }


        } catch (FileNotFoundException e) {
            System.out.println("File Not found");
        }

        // Use stored value for output stream
        System.setOut(console);
        System.out.println("This will be written on the console!");

    }


}

class Triple {

    private int docNum;//Stores the document number
    private ArrayList<Word> wordsWithCounts;//Stores the tuples containing words and their counts
    private int category;//Stores the category that this doc belongs to

    public Triple(int docNum, int category) {
        this.docNum = docNum;
        this.category = category;
        wordsWithCounts = new ArrayList<>();
    }

    public void addWordToDoc(Word w) {
        wordsWithCounts.add(w);
    }

    public ArrayList<Word> getWordsWithCounts() {
        return wordsWithCounts;
    }

    public int getCategory() {
        return category;
    }

}

class Word {

    private int wordID;//Stores the word ID
    private int count;//Stores the number of times the word "wordIdx" occurs in document "docNum"

    public Word(int wordID, int count) {
        this.wordID = wordID;
        this.count = count;
    }

    public int getWordID() {
        return wordID;
    }

    public int getCount() {
        return count;
    }

}