// LLVM Includes
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

// using namespace std;
using namespace llvm;
static LLVMContext Context;

#define DEBUG_TYPE "MiniSLP"

static const char lv_name[] = "MiniSLP pass";

namespace {

struct Node { // Node that will hold the instructions of an Isomorphic tree

  std::vector<Value *> ScalarInstructions;

  bool canBeVectorizable;
  std::string whyCantBeVectorizable;

  int scalarCost;
  int vecCost;
  int extractOverheadCost;
  int insertOverheadCost;

  // Next and previous nodes
  std::vector<Node *> pointsToNodes;
  Node *isPointedByNode;

  Node() {
    resetCost();
    isPointedByNode = nullptr;
    canBeVectorizable = true;
    whyCantBeVectorizable =
        "*failed to get the reason (Isn't it vectorizable?)*";
  }

  void setAsCantBeVectorizable(std::string reason) {
    canBeVectorizable = false;
    whyCantBeVectorizable = reason;
  }

  void resetCost() {
    scalarCost = 0;
    vecCost = 0;
    extractOverheadCost = 0;
    insertOverheadCost = 0;
  }

  int getOverheadCost() { return extractOverheadCost + insertOverheadCost; }

  int getNodeCost() {

    return vecCost + extractOverheadCost + insertOverheadCost - scalarCost;
  }

  void insertInst(Value *it) { ScalarInstructions.push_back(it); }

  bool isInstInNode(Instruction *I) {

    for (unsigned i = 0; i < ScalarInstructions.size(); i++) {

      if (ScalarInstructions[i] == I)
        return true;
    }

    return false;
  }

  bool isLeaf() { return pointsToNodes.empty(); }
};

struct IsoTree {

  Node *seedNode;
  std::vector<Node *> LeafNodes;
  int treeCost;
  int vectorWidth;
  Type *biggestType;

  IsoTree() {
    treeCost = 0;
    vectorWidth = 0;
    biggestType = nullptr;
  }

  std::vector<Node *> getLastVectorizableNodes() {
    std::vector<Node *> lastVectorizableNodes;

    for (unsigned i = 0; i < LeafNodes.size(); i++) {
      Node *leafNodeIsPointedBy = LeafNodes[i]->isPointedByNode;
      if (leafNodeIsPointedBy->canBeVectorizable) {
        // Check if the node is already in the vector
        if (std::find(lastVectorizableNodes.begin(),
                      lastVectorizableNodes.end(),
                      leafNodeIsPointedBy) == lastVectorizableNodes.end()) {
          lastVectorizableNodes.push_back(leafNodeIsPointedBy);
        }
      }
    }

    return lastVectorizableNodes;
  }

  std::vector<Node *> getLeafNodes() {
    std::vector<Node *> leafNodes;
    getLeafNodes(seedNode, leafNodes);
    return leafNodes;
  }

  void getLeafNodes(Node *node, std::vector<Node *> &leafNodes) {
    if (node->isLeaf()) {
      leafNodes.push_back(node);
    }

    for (unsigned i = 0; i < node->pointsToNodes.size(); i++) {
      getLeafNodes(node->pointsToNodes[i], leafNodes);
    }
  }

  int getNumOfNodes(Node *currentNode = nullptr, int nodeCounter = 0) {
    if (currentNode == nullptr) {
      currentNode = seedNode;
    }
    for (unsigned i = 0; i < currentNode->pointsToNodes.size(); i++) {
      nodeCounter = getNumOfNodes(currentNode->pointsToNodes[i], nodeCounter);
    }
    return nodeCounter + 1;
  }

  Node *getSeedNode() { return seedNode; }

  bool isInstructionInVectorizableTree(Value *it, Node *currentNode = nullptr) {

    if (currentNode == nullptr) {
      currentNode = getSeedNode();
    }

    if (currentNode->canBeVectorizable) {
      for (unsigned i = 0; i < currentNode->ScalarInstructions.size(); i++) {
        if (currentNode->ScalarInstructions[i] == it)
          return true;
      }
    }

    for (unsigned i = 0; i < currentNode->pointsToNodes.size(); i++) {

      if (isInstructionInVectorizableTree(it, currentNode->pointsToNodes[i]))
        return true;
    }

    return false;
  }

  Node *getNodeOfInstruction(Value *it, Node *currentNode = nullptr) {

    if (currentNode == nullptr) {
      currentNode = getSeedNode();
    }
    for (unsigned i = 0; i < currentNode->ScalarInstructions.size(); i++) {

      if (currentNode->ScalarInstructions[i] == it)
        return currentNode;
    }

    for (unsigned i = 0; i < currentNode->pointsToNodes.size(); i++) {

      return (getNodeOfInstruction(it, currentNode->pointsToNodes[i]));
    }

    return nullptr;
  }
};

struct MiniSLP : public FunctionPass {
  static char ID;
  MiniSLP() : FunctionPass(ID) {}

  TargetTransformInfo *TTI;
  TargetLibraryInfoWrapperPass *TLIP;
  TargetLibraryInfo *TLI;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    errs() << "\n\nStarting Mini-SLP Pass...";

    TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
    TLI = TLIP ? &TLIP->getTLI() : nullptr;

    bool updated = false;

    errs() << "\n<=======> Analising function '" << (F.getName())
           << "' <=======>\n";
    int BBCounter = 1;

    for (auto bb = F.begin(); bb != F.end();
         bb++) { // For each basic block in the function
      errs() << "\n\n<-------> BASIC BLOCK #" << BBCounter << " <------->\n";

      std::vector<StoreInst *> storeInsts;
      for (auto inst = bb->begin(); inst != bb->end();
           inst++) { // For each instruction in the basic block

        if (StoreInst *si = dyn_cast<StoreInst>(&*inst)) {

          Type *type = si->getPointerOperandType()->getPointerElementType();

          if (type->isIntegerTy()) {
            storeInsts.push_back(si);
          }
        }
      }

      errs() << "\nTrees found by int stores:";
      for (unsigned i = 0; i < storeInsts.size(); i++) {
        showTree(storeInsts[i]);
      }

      findIsomorphicTreesAndCastThem(storeInsts);

      BBCounter++;
    }

    errs() << "\n";

    return updated;
  }

  void findIsomorphicTreesAndCastThem(std::vector<StoreInst *> &storeInsts) {

    std::vector<IsoTree *> isoTrees;

    isoTrees = generateIsomorphicTrees(storeInsts);

    // Print the isomorphic tress formed
    errs() << "\n\n=== Printing isomorphic trees formed ===";
    for (unsigned i = 0; i < isoTrees.size(); i++) {
      showIsoTree(isoTrees[i]);
      errs() << "\n\n  ~> Biggest type: " << *isoTrees[i]->biggestType;
      errs() << "\n  ~> Tree cost: " << isoTrees[i]->treeCost << "\n";
    }

    // Casting trees
    errs() << "\n\n=== Casting isomorphic trees to the biggest type ===";
    for (unsigned i = 0; i < isoTrees.size(); i++) {
      isoTrees[i] = castToBiggestType(isoTrees[i]);

      /**
       * TODO: Print IsoTree and update IsoTree
       **/
      // //Print the new tree
      // showIsoTree(isoTrees[i]);
      // errs() << "\n\n  ~> Tree cost: " << isoTrees[i]->treeCost << "\n";
      // errs() << "\n\n";
      /**
       * END TODO
       **/
    }
  }

  /**
   * TODO:
   **/
  IsoTree *castToBiggestType(IsoTree *isoTree) {
    // // CreateExtInsts(isoTree); //We don't change the last nodes (they're are
    // // not vectorizable), so create a SExt or ZExt std::vector<Instruction*>
    // // instToRecreate = FindInstructionsToRecreate(isoTree);
    // // RecreateInsts(instToRecreate, isoTree->biggestType);
    // // std::vector<Value*> newSeeds =
    // // CreateTruncateInstsAndGetNewSeeds(isoTree); //We don't change the first
    // // nodes (they're are not vectorizable), so create a truncate to their
    // // original type

    // // IsoTree* newIsoTree = createIsoTree(newSeeds, true);
    // // setTreeCost(newIsoTree);

    // // return newIsoTree;

	  //Cria os extends e transforma a última instrução dos últimos nó vetorizados como vetor
    CriaExt(isoTree);
	
    return isoTree;
  }
  void CriaExt(IsoTree *isoTree) {
    auto *vectorTy = VectorType::get(isoTree->biggestType, isoTree->vectorWidth);
    std::vector<Value*> vecOperands;
    std::vector<Value*> instVec;
    std::vector<Instruction*> delInsts;
    std::vector<Node *> lastNodes = isoTree->getLastVectorizableNodes();

	  //Para cada ultimo node
    for (Node *node : lastNodes) {
      //Pegue a última instrução e crie um vetor do tipo dela
      Instruction *lastInst = dyn_cast<Instruction>(node->ScalarInstructions.back());	
      //E faça um loop que percorra cada operando
      for (unsigned i = 0; i < lastInst->getNumOperands(); i++) {				
        Value *vec = UndefValue::get(vectorTy);
        IRBuilder<> builder(lastInst);
        //De uma instrução 
        for (Value* inst : node->ScalarInstructions) {
          Instruction *nodeInst = dyn_cast<Instruction>(inst);

          //Crie um Ext quando necessário para o operando i de todas instruções escalares
          Value *operand = nodeInst->getOperand(i);
          Value *newOperand;
          if (isIntegerSigned(operand)) {
            newOperand = builder.CreateSExt(operand, isoTree->biggestType);
          } else {
            newOperand = builder.CreateZExt(operand, isoTree->biggestType);
          }

          instVec.push_back(newOperand);
        }
        for(unsigned i = 0; i < instVec.size(); i++){
          //Armazene-o em um vetor (ext ou instrução)
          vec = builder.CreateInsertElement(vec, instVec[i], i);
        }
        //Salve as intruções de vetores de operandos salvas
        vecOperands.push_back(vec);
        instVec.clear();
      }
      //Gera uma instrução vetor para o node com sua primeira instrução
      lastInst->mutateType(vectorTy);
      lastInst->setOperand(0, vecOperands[0]);
      lastInst->setOperand(1, vecOperands[1]);


      //Atualiza a sub arvore do node folha
      updateSubTree(isoTree, node, lastInst, delInsts);

      // Remove as instruções não usadas
      for(Instruction* i : delInsts){
        i->eraseFromParent();
      }
    }
  }
  void updateSubTree(IsoTree* isoTree, Node* node, Instruction* lastInst, std::vector<Instruction*> delInsts){
    auto *vectorTy = VectorType::get(isoTree->biggestType, isoTree->vectorWidth);
    //Se o nó é apontado pelo Store
    if(node->isPointedByNode == isoTree->seedNode){
      Node* nodePrevious = node->isPointedByNode;

      //Remove as instruções não usadas mais no node
      for(unsigned i = 0; i < node->ScalarInstructions.size() - 1; i++){
        Instruction* del = dyn_cast<Instruction>(node->ScalarInstructions[i]);
        delInsts.push_back(del);
      }
      //Atualiza o operando dos stores
      IRBuilder<> builder(lastInst->getNextNode());
      Instruction* lastExtract;
      for(unsigned i = 0; i < nodePrevious->ScalarInstructions.size(); i++){
        Value* extract = builder.CreateExtractElement(lastInst, i);
        StoreInst* store = dyn_cast<StoreInst>(nodePrevious->ScalarInstructions[i]);
        
        Type *typeSi = store->getPointerOperandType()->getPointerElementType();
        Value *newOperand = builder.CreateTrunc(extract, typeSi);
        store->setOperand(0, newOperand);

        lastExtract = dyn_cast<Instruction>(extract);
      }
      //Move os stores para o final
      for(unsigned i = 0; i < nodePrevious->ScalarInstructions.size(); i++){
        StoreInst* store = dyn_cast<StoreInst>(nodePrevious->ScalarInstructions[i]);
        store->moveAfter(lastExtract);
        nodePrevious->ScalarInstructions[i] = store;
      }
    }
    else {
      Node* nodePrevious = node->isPointedByNode;
      Instruction* newLastInst = dyn_cast<Instruction>(nodePrevious->ScalarInstructions.back());
      newLastInst->mutateType(vectorTy);
      if(newLastInst->getType() == vectorTy){
        newLastInst->setOperand(1, lastInst);
      }
      else {
        newLastInst->setOperand(0, lastInst);
      }

      //Remove as instruções não usadas mais no node
      for(unsigned i = 0; i < node->ScalarInstructions.size() - 1; i++){
        Instruction* del = dyn_cast<Instruction>(node->ScalarInstructions[i]);
        delInsts.push_back(del);
      }

      updateSubTree(isoTree, nodePrevious, newLastInst, delInsts);
    }
  }
  /**
   * END TODO:
   **/

  std::vector<IsoTree *>
  generateIsomorphicTrees(std::vector<StoreInst *> &storeInsts) {
    std::vector<IsoTree *> IsomorphicTrees;
    errs() << "\n\n=== Finding isomorphic trees ===";
    errs() << "\nReceived " << storeInsts.size() << " trees.\n\n";

    std::vector<std::vector<Value *>> vectorOfBestStores =
        findBestCombinationOfStoreInsts(storeInsts);
    for (unsigned i = 0; i < vectorOfBestStores.size(); i++) {
      IsoTree *isoTree = createIsoTree(vectorOfBestStores[i], true);
      setTreeCost(isoTree);
      IsomorphicTrees.push_back(isoTree);
    }

    return IsomorphicTrees;
  }

  // Find the best combination of stores to create a Isomorphic Tree, currently
  // it only analyses pairs of stores
  std::vector<std::vector<Value *>>
  findBestCombinationOfStoreInsts(std::vector<StoreInst *> &storeInsts) {

    std::vector<std::vector<Value *>> vectorOfBestStores;

    while (!storeInsts.empty()) {
      std::vector<Value *> bestStores;
      int higherNumberOfCommonNodes = -1;
      int bestPairIndex = -1;
      Value *firstStoreInst =
          storeInsts[0]; // Get the first store from the std::vector

      // Compare the first store tree with all others and get the best pair
      for (unsigned k = 1; k < storeInsts.size(); k++) {
        Value *kStoreInst = storeInsts[k];
        int numberOfCommonNodes =
            getNumberOfCommnNodes({firstStoreInst, kStoreInst});

        if (higherNumberOfCommonNodes == -1 ||
            numberOfCommonNodes > higherNumberOfCommonNodes) {

          higherNumberOfCommonNodes = numberOfCommonNodes;
          bestStores = {firstStoreInst, kStoreInst};
          bestPairIndex = k;
        }
      }

      // Compare if the first tree is isomorphic to any other:
      if (higherNumberOfCommonNodes > 0) {

        vectorOfBestStores.push_back(bestStores);

        errs() << "\nIsomorphic tree generated by grouping "
               << bestStores.size() << " trees:\n";

        for (unsigned i = 0; i < bestStores.size(); i++) {
          errs() << *bestStores[i] << "\n";
        }
        errs() << "Common nodes: " << higherNumberOfCommonNodes << "\n";

        storeInsts.erase(storeInsts.begin() + bestPairIndex);
      } else if (higherNumberOfCommonNodes == 0) {
        errs() << "The tree of store \"" << *firstStoreInst
               << " \" is not isomorphic with any other tree.\n";
      } else {
        errs() << "No remaining trees to compare the store \""
               << *firstStoreInst << " \"";
      }

      storeInsts.erase(storeInsts.begin());
    }

    return vectorOfBestStores;
  }

  // Create a IsoTree object from a std::vector of StoreInsts
  IsoTree *createIsoTree(std::vector<Value *> &instNode, bool isSeed,
                         IsoTree *tree = new IsoTree(),
                         Node *previousNode = nullptr) {

    Node *newNode = new Node();
    if (previousNode != nullptr) {
      previousNode->pointsToNodes.push_back(newNode);
      newNode->isPointedByNode = previousNode;
    }
    // Insert the instructions in the node
    for (unsigned i = 0; i < instNode.size(); i++) {
      newNode->insertInst(instNode[i]);
    }

    if (checkIfAllValuesAreInstructions(instNode)) {

      if (checkIfAllInstructionsHaveSameOpCode(instNode)) {

        if (checkIfAllInstructionsAreIntegers(instNode)) {
          tree->biggestType = getBiggestInteger(instNode);

          // If it's a Load Instrunction, then it's a leaf
          if (isa<LoadInst>(instNode[0])) {
            // And we will not vectorize it
            newNode->setAsCantBeVectorizable(
                "it's a node of Load Instructions. (final of the tree)");
          }

          // If it's a Store Instrunction, then we need just the ValueOperand,
          // and need to set up some variables in tree
          else if (isa<StoreInst>(instNode[0])) {
            tree->seedNode = newNode;
            newNode->setAsCantBeVectorizable(
                "it's a node of Store Instructions (beggining of the tree).");
            tree->vectorWidth = instNode.size();

            std::vector<Value *> operandInstrucs;

            for (unsigned j = 0; j < instNode.size(); j++) {

              StoreInst *si = dyn_cast<StoreInst>(instNode[j]);
              operandInstrucs.push_back(si->getValueOperand());
            }

            tree = createIsoTree(operandInstrucs, false, tree, newNode);
          }
          // If it's another instruction, we continue the tree
          else {
            unsigned numOps =
                dyn_cast<Instruction>(instNode[0])->getNumOperands();

            for (unsigned i = 0; i < numOps; i++) {
              std::vector<Value *> operandInstrucs;

              for (unsigned j = 0; j < instNode.size(); j++) {

                Value *vOp = dyn_cast<Instruction>(instNode[j])->getOperand(i);
                operandInstrucs.push_back(vOp);
              }

              tree = createIsoTree(operandInstrucs, false, tree, newNode);
            }
          }
        } else {
          newNode->setAsCantBeVectorizable("Not all instructions are integers");
        }
      } else {
        newNode->setAsCantBeVectorizable(
            "not all instructions in the node have the same Opcode.");

        if (isSeed) {
          tree->biggestType = getBiggestInteger(instNode);

          tree->seedNode = newNode;
          tree->vectorWidth = instNode.size();

          std::vector<Value *> operandInstrucs;
          for (unsigned j = 0; j < instNode.size(); j++) {

            if (StoreInst *si = dyn_cast<StoreInst>(instNode[j])) {
              operandInstrucs.push_back(si->getValueOperand());
            } else if (TruncInst *ti = dyn_cast<TruncInst>(instNode[j])) {
              operandInstrucs.push_back(ti->getOperand(0));
            }
          }

          tree = createIsoTree(operandInstrucs, false, tree, newNode);
        }
      }
    } else {
      newNode->setAsCantBeVectorizable(
          "not all elements of the node are instructions.");
    }

    if (newNode->pointsToNodes.empty()) {
      tree->LeafNodes.push_back(newNode);
    }

    return tree;
  }

  bool checkIfAllValuesAreInstructions(const std::vector<Value *> &values) {
    for (unsigned i = 0; i < values.size(); i++) {
      if (!isa<Instruction>(values[i])) {
        return false;
      }
    }

    return true;
  }

  bool
  checkIfAllInstructionsHaveSameOpCode(const std::vector<Value *> &instrucs) {
    for (unsigned i = 1; i < instrucs.size(); i++) {
      if (dyn_cast<Instruction>(instrucs[i])->getOpcode() !=
          dyn_cast<Instruction>(instrucs[0])->getOpcode()) {

        return false;
      }
    }

    return true;
  }

  bool checkIfAllInstructionsAreIntegers(const std::vector<Value *> &instrucs) {
    for (unsigned i = 1; i < instrucs.size(); i++) {
      Type *typei = getValueType(instrucs[i]);
      if (!typei->isIntegerTy()) {
        return false;
      }
    }

    return true;
  }

  bool
  checkIfAllInstructionsHaveTheSameType(const std::vector<Value *> &instrucs) {
    if (!instrucs.empty()) {
      Type *type0 = getValueType(instrucs[0]);
      for (unsigned i = 1; i < instrucs.size(); i++) {
        Type *typei = getValueType(instrucs[i]);
        if (type0 != typei) {
          return false;
        }
      }
    }

    return true;
  }

  Type *getBiggestInteger(const std::vector<Value *> &instrucs) {
    Type *biggestType = nullptr;

    if (!instrucs.empty()) {
      biggestType = getValueType(instrucs[0]);
      for (unsigned i = 1; i < instrucs.size(); i++) {
        Type *typei = getValueType(instrucs[i]);
        if (typei->getIntegerBitWidth() > biggestType->getIntegerBitWidth()) {
          biggestType = typei;
        }
      }
    }

    return biggestType;
  }

  // Get the number of common nodes between trees
  int getNumberOfCommnNodes(const std::vector<Value *> &instNode,
                            int commonNodesSoFar = 0) {

    if (checkIfAllValuesAreInstructions(instNode) &&
        checkIfAllInstructionsHaveSameOpCode(instNode) &&
        checkIfAllInstructionsAreIntegers(instNode)) {

      // If it's a Load Instrunction, then it's a leaf (right now we will not
      // vectorize loads)
      if (isa<LoadInst>(instNode[0])) {
      }

      // If it's a Store Instrunction, then we need just the ValueOperand
      else if (isa<StoreInst>(instNode[0])) {

        std::vector<Value *> OperandInsts;

        for (unsigned j = 0; j < instNode.size(); j++) {
          StoreInst *si = dyn_cast<StoreInst>(instNode[j]);
          OperandInsts.push_back(si->getValueOperand());
        }

        commonNodesSoFar =
            getNumberOfCommnNodes(OperandInsts, commonNodesSoFar);
      }

      else {
        Instruction *currentInstruction = dyn_cast<Instruction>(instNode[0]);
        unsigned numOps = currentInstruction->getNumOperands();
        commonNodesSoFar++;

        for (unsigned i = 0; i < numOps; i++) {
          std::vector<Value *> OperandInsts;
          for (unsigned j = 0; j < instNode.size(); j++) {

            Value *vOp = dyn_cast<Instruction>(instNode[j])->getOperand(i);
            OperandInsts.push_back(vOp);
          }

          commonNodesSoFar =
              getNumberOfCommnNodes(OperandInsts, commonNodesSoFar);
        }
      }
    }

    return commonNodesSoFar;
  }

  void CreateExtInsts(IsoTree *isoTree) {

    std::vector<Node *> lastVectorizableNodes =
        isoTree->getLastVectorizableNodes();
    std::vector<Value *> newOperands;

    for (unsigned nodeIndex = 0; nodeIndex < lastVectorizableNodes.size();
         nodeIndex++) { // For every last vectorizable node
      Node *node = lastVectorizableNodes[nodeIndex];

      for (unsigned scalarIndex = 0;
           scalarIndex < node->ScalarInstructions.size();
           scalarIndex++) { // We go through the instructions
        Instruction *nodeInst =
            dyn_cast<Instruction>(node->ScalarInstructions[scalarIndex]);
        IRBuilder<> builder(nodeInst);

        for (unsigned operandIndex = 0;
             operandIndex < nodeInst->getNumOperands();
             operandIndex++) { // And cast its operands to the biggest type
          Value *operand = nodeInst->getOperand(operandIndex);
          Value *newOperand;
          if (isIntegerSigned(operand)) {
            newOperand = builder.CreateSExt(operand, isoTree->biggestType);
          } else {
            newOperand = builder.CreateZExt(operand, isoTree->biggestType);
          }
          nodeInst->setOperand(operandIndex, newOperand);
        }
      }
    }
  }

  // LLVM doesn't differenciate signed integers from unsigned ones, only the
  // instructions that use them, but since we need to create a cast, we need to
  // know if a int is or not signed in order to use SExt or ZExt instructions.
  // Currently it only verifies by looking the users tree of the value and
  // checking if 'udiv' appears.
  bool isIntegerSigned(Value *value) {
    for (auto user : value->users()) {

      if (isa<UDivOperator>(value)) {
        return false;
      } else {
        return isIntegerSigned(user);
      }
    }

    return true;
  }

  std::vector<Instruction *> FindInstructionsToRecreate(IsoTree *isoTree) {
    std::vector<Instruction *> instToRecreate;
    return FindInstructionsToRecreate(
        isoTree, isoTree->getLastVectorizableNodes(), instToRecreate);
  }

  std::vector<Instruction *>
  FindInstructionsToRecreate(IsoTree *isoTree,
                             const std::vector<Node *> nodesToBeAnalyzed,
                             std::vector<Instruction *> &instToRecreate) {
    for (unsigned i = 0; i < nodesToBeAnalyzed.size(); i++) {
      Node *node = nodesToBeAnalyzed[i];

      if (node->canBeVectorizable) {
        for (unsigned k = 0; k < node->ScalarInstructions.size(); k++) {
          instToRecreate.push_back(
              dyn_cast<Instruction>(node->ScalarInstructions[k]));
        }
        FindInstructionsToRecreate(isoTree, {node->isPointedByNode},
                                   instToRecreate);
      }
    }

    return instToRecreate;
  }

  // Recreate the instructions using the new casted operands
  // Currently we are only working with BinaryOps, so we're justing mutating the
  // type
  void RecreateInsts(const std::vector<Instruction *> &insts, Type *newType) {
    for (unsigned i = 0; i < insts.size(); i++) {
      if (BinaryOperator *op = dyn_cast<BinaryOperator>(insts[i])) {
        op->mutateType(newType);
      }
    }
  }

  std::vector<Value *> CreateTruncateInstsAndGetNewSeeds(IsoTree *isoTree) {
    std::vector<Value *> newSeeds;
    Node *node = isoTree->getSeedNode();

    for (unsigned i = 0; i < node->ScalarInstructions.size(); i++) {
      StoreInst *si = dyn_cast<StoreInst>(node->ScalarInstructions[i]);
      IRBuilder<> builder(si);

      Value *siOperand = si->getOperand(0);
      Type *typeSi = si->getPointerOperandType()->getPointerElementType();
      Value *newOperand = builder.CreateTrunc(siOperand, typeSi);

      if (siOperand == newOperand) { // If the instruction didn't change, it's
                                     // already a tree of the bigges type
        newSeeds.push_back(si);
      } else {
        newSeeds.push_back(newOperand);
      }

      si->setOperand(0, newOperand);
    }

    return newSeeds;
  }

  Type *getValueType(Value *V) {

    if (StoreInst *si = dyn_cast<StoreInst>(V)) {
      return si->getPointerOperandType()->getPointerElementType();
    } else {
      return V->getType();
    }
  }

  // Recursively show a tree receiving a store instruction as parameter
  void showTree(StoreInst *inst) {
    std::vector<Value *> instrucs = {inst};
    showTreeHelper(instrucs);
  }

  void showTreeHelper(const std::vector<Value *> &instrucs, int space = 0,
                      std::vector<bool> writePipe = {}) {

    for (unsigned i = 0; i < instrucs.size(); i++) {
      if (i == 0)
        writePipe.push_back(true);
      writePipe[writePipe.size() - 1] = (i != instrucs.size() - 1);

      Instruction *si = dyn_cast<Instruction>(instrucs[i]);

      std::vector<Value *> ops;

      errs() << "\n";
      if (si->getOpcode() == Instruction::Store)
        errs() << "\n";

      for (int k = 0, j = 0; k < space; k++) {
        if (k % 4 == 0) {
          if (writePipe[j + 1] || k == space - 4) {
            errs() << "|";
          } else
            errs() << " ";
          j++;
        } else
          errs() << " ";
      }
      errs() << "\n";
      for (int k = 0, j = 0; k < space; k++) {
        if (k == space - 1)
          errs() << ">";
        else if (k % 4 == 0) {
          if (writePipe[j + 1] || k == space - 4) {
            errs() << "|";
          } else
            errs() << " ";

          j++;
        } else if (k > space - 4)
          errs() << "-";
        else
          errs() << " ";
      }

      errs() << si->getOpcodeName();

      Value *vI = dyn_cast<Value>(si);
      Type *type = getValueType(vI);
      errs() << " (" << *type;
      if (type->isIntegerTy())
        errs() << " integer";
      errs() << ") ~~>" << *si << " ~~> ("
             << "Scalar Cost = " << GetScalarCost(si, TTI, TLI, Context) << ")";

      for (unsigned k = 0; k < si->getNumOperands(); k++) {
        Value *vOp = si->getOperand(k);
        if (isa<Instruction>(vOp)) {
          ops.push_back(vOp);
        }
      }

      showTreeHelper(ops, space + 4, writePipe);
    }
  }

  void setTreeCost(IsoTree *tree, Node *node = nullptr,
                   bool firstIteration = true) {

    if (firstIteration) {
      firstIteration = false;
      tree->treeCost = 0;
      node = tree->getSeedNode();
    }

    node->resetCost();

    if (node->canBeVectorizable) {
      bool extractAlreadyCalculated =
          false; //<- We only calculate the extract overhead once
      bool insertAlreadyCalculated =
          false; //<- We only calculate the insert overhead once

      for (unsigned i = 0; i < node->ScalarInstructions.size();
           i++) { // For each instruction in the node

        Instruction *nodeInstruction =
            dyn_cast<Instruction>(node->ScalarInstructions[i]);

        node->scalarCost += GetScalarCost(nodeInstruction, TTI, TLI, Context);

        if (!extractAlreadyCalculated) {
          // Look for extract overheads
          for (auto user : node->ScalarInstructions[i]
                               ->users()) { // For each user of this instruction

            if (Instruction *userInstruction = dyn_cast<Instruction>(
                    user)) { // Check if it is another instruction

              if (!tree->isInstructionInVectorizableTree(
                      userInstruction)) // And check if this user is in the tree
                                        // and is vectorizable, if not, we have
                                        // an extract overhead
              {
                int overhead = GetVectorExtractCost(
                    nodeInstruction, tree->vectorWidth, TTI, TLI, Context);
                node->extractOverheadCost += overhead;
                extractAlreadyCalculated = true;

                break;
              }
            }
          }
        }

        if (!insertAlreadyCalculated) {
          // Look for insert overheads
          for (unsigned k = 0; k < nodeInstruction->getNumOperands(); k++) {
            Value *Op = nodeInstruction->getOperand(k);
            if (Instruction *opInstruction = dyn_cast<Instruction>(
                    Op)) { // Check if it is another instruction

              if (!tree->isInstructionInVectorizableTree(opInstruction)) {

                int overhead = GetVectorInsertCost(
                    nodeInstruction, tree->vectorWidth, TTI, TLI, Context);
                node->insertOverheadCost += overhead;
                insertAlreadyCalculated = true;

                break;
              }
            }
          }
        }
      }

      node->vecCost =
          GetVectorCost(dyn_cast<Instruction>(node->ScalarInstructions[0]),
                        tree->vectorWidth, TTI, TLI, Context);
    }

    tree->treeCost += node->getNodeCost();

    for (unsigned i = 0; i < node->pointsToNodes.size(); i++) {
      setTreeCost(tree, node->pointsToNodes[i], false);
    }
  }

  // Print an isomorphic tree
  void showIsoTree(IsoTree *tree, Node *currentNode = nullptr, int space = 0,
                   std::vector<bool> writePipe = {}) {

    errs() << "\n";
    int width = tree->vectorWidth;
    if (width == -1) {
      errs() << "Tree is not initialized.";
    } else {
      if (currentNode == nullptr) {
        currentNode = tree->getSeedNode();
      }

      for (int i = 0; i < width; i++) {
        if (i != width - 1) {
          for (int k = 0, j = 0; k < space; k++) {
            if (k % 4 == 0) {
              if (writePipe[j] || k == space - 4) {
                errs() << "|";
              } else
                errs() << " ";
              j++;
            } else
              errs() << " ";
          }
        }

        errs() << "\n";
        for (int k = 0, j = 0; k < space; k++) {
          if (k == space - 1)
            errs() << ">";
          else if (k % 4 == 0) {
            if (writePipe[j] || k == space - 4) {
              errs() << "|";
            } else
              errs() << " ";

            j++;
          } else if (k > space - 4)
            errs() << "-";
          else
            errs() << " ";
        }
        if (Instruction *I =
                dyn_cast<Instruction>(currentNode->ScalarInstructions[i])) {
          errs() << I->getOpcodeName() << " ~~>" << *I;
        } else {

          errs() << *currentNode->ScalarInstructions[i];
        }
      }

      errs() << " ~> ";
      if (currentNode->canBeVectorizable) {
        errs() << "Node cost: " << currentNode->getNodeCost();
        errs() << " (Vector " << currentNode->vecCost;
        errs() << " + " << currentNode->getOverheadCost();
        if (currentNode->insertOverheadCost > 0) {

          errs() << " Insert";
        }
        if (currentNode->extractOverheadCost > 0) {

          errs() << " Extract";
        }
        errs() << " Overhead";
        errs() << " - " << currentNode->scalarCost << " Scalar)";

        if (checkIfAllInstructionsHaveTheSameType(
                currentNode->ScalarInstructions)) {
          errs() << " - Vectorizable.";
        } else {
          errs() << " - Non-vectorizable, but can be if we cast it to "
                 << *tree->biggestType << ".";
        }
      } else {

        errs() << "Non-vectorizable and can't be at all. (Because " +
                      currentNode->whyCantBeVectorizable + ")";
      }

      for (unsigned i = 0; i < currentNode->pointsToNodes.size(); i++) {

        if (i == 0) {
          writePipe.push_back(false);
        }

        writePipe[writePipe.size() - 1] =
            (i != currentNode->pointsToNodes.size() - 1);

        showIsoTree(tree, currentNode->pointsToNodes[i], space + 4, writePipe);
      }
    }
  }

  static int GetScalarCost(Instruction *I, TargetTransformInfo *TTI,
                           TargetLibraryInfo *TLI, LLVMContext &Context) {

    Type *ScalarTy = I->getType();
    if (isa<StoreInst>(I)) {
      ScalarTy = cast<StoreInst>(I)->getValueOperand()->getType();
    } else if (isa<CmpInst>(I)) {
      ScalarTy = cast<CmpInst>(I)->getOperand(0)->getType();
    }

    switch (I->getOpcode()) {
    case Instruction::PHI:
      return 0;
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::BitCast: {
      Type *SrcTy = I->getOperand(0)->getType();
      return TTI->getCastInstrCost(I->getOpcode(), ScalarTy, SrcTy, I);
    }
    case Instruction::FCmp:
    case Instruction::ICmp:
    case Instruction::Select: {
      // Calculate the cost of this instruction.
      return TTI->getCmpSelInstrCost(I->getOpcode(), ScalarTy,
                                     IntegerType::get(Context, 1), I);
    }
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      // Certain instructions can be cheaper to vectorize if they have a
      // constant second std::vector operand.
      TargetTransformInfo::OperandValueKind Op1VK =
          TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Op2VK =
          TargetTransformInfo::OK_UniformConstantValue;
      TargetTransformInfo::OperandValueProperties Op1VP =
          TargetTransformInfo::OP_None;
      TargetTransformInfo::OperandValueProperties Op2VP =
          TargetTransformInfo::OP_PowerOf2;

      ConstantInt *CInt = dyn_cast<ConstantInt>(I->getOperand(0));
      if (!CInt) {
        Op1VK = TargetTransformInfo::OK_AnyValue;
        Op1VP = TargetTransformInfo::OP_None;
      } else {
        Op1VK = TargetTransformInfo::OK_UniformConstantValue;
        if (CInt->getValue().isPowerOf2())
          Op1VP = TargetTransformInfo::OP_PowerOf2;
        else
          Op1VP = TargetTransformInfo::OP_None;
      }

      CInt = dyn_cast<ConstantInt>(I->getOperand(1));
      if (!CInt) {
        Op2VK = TargetTransformInfo::OK_AnyValue;
        Op2VP = TargetTransformInfo::OP_None;
      } else {
        Op2VK = TargetTransformInfo::OK_UniformConstantValue;
        if (CInt->getValue().isPowerOf2())
          Op2VP = TargetTransformInfo::OP_PowerOf2;
        else
          Op2VP = TargetTransformInfo::OP_None;
      }

      SmallVector<const Value *, 4> Operands(I->operand_values());
      return TTI->getArithmeticInstrCost(I->getOpcode(), ScalarTy, Op1VK, Op2VK,
                                         Op1VP, Op2VP, Operands);
    }
    case Instruction::GetElementPtr: {
      TargetTransformInfo::OperandValueKind Op1VK =
          TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Op2VK =
          TargetTransformInfo::OK_UniformConstantValue;

      return TTI->getArithmeticInstrCost(Instruction::Add, ScalarTy, Op1VK,
                                         Op2VK);
    }
    case Instruction::Load: {
      // Cost of wide load - cost of scalar loads.
      unsigned alignment = cast<LoadInst>(I)->getAlignment();
      return TTI->getMemoryOpCost(Instruction::Load, ScalarTy, alignment, 0, I);
    }
    case Instruction::Store: {
      // We know that we can merge the stores. Calculate the cost.
      unsigned alignment = cast<StoreInst>(I)->getAlignment();
      Type *SrcTy = cast<StoreInst>(I)->getValueOperand()->getType();
      return TTI->getMemoryOpCost(Instruction::Store, SrcTy, alignment, 0, I);
    }
    case Instruction::Call: {
      CallInst *CI = cast<CallInst>(I);
      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);

      // Calculate the cost of the scalar and std::vector calls.
      SmallVector<Type *, 4> ScalarTys;
      for (unsigned op = 0, opc = CI->getNumArgOperands(); op != opc; ++op)
        ScalarTys.push_back(CI->getArgOperand(op)->getType());

      FastMathFlags FMF;
      if (auto *FPMO = dyn_cast<FPMathOperator>(CI))
        FMF = FPMO->getFastMathFlags();

      return TTI->getIntrinsicInstrCost(ID, ScalarTy, ScalarTys, FMF);
    }
    default:
      return 0;
    }
  }

  static int GetVectorInsertCost(Value *V, unsigned Count,
                                 TargetTransformInfo *TTI,
                                 TargetLibraryInfo *TLI, LLVMContext &Context) {
    Type *ScalarTy = V->getType();

    if (!VectorType::isValidElementType(ScalarTy))
      return 0;
    VectorType *VecTy = VectorType::get(ScalarTy, Count);

    int Cost = 0;
    for (unsigned i = 0; i < Count; i++)
      Cost += TTI->getVectorInstrCost(Instruction::InsertElement, VecTy, i);
    return Cost;
  }

  static int GetVectorExtractCost(Value *V, unsigned Count,
                                  TargetTransformInfo *TTI,
                                  TargetLibraryInfo *TLI,
                                  LLVMContext &Context) {
    Type *ScalarTy = V->getType();

    if (!VectorType::isValidElementType(ScalarTy))
      return 0;
    VectorType *VecTy = VectorType::get(ScalarTy, Count);

    int Cost = 0;
    for (unsigned i = 0; i < Count; i++)
      Cost += TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy, i);
    return Cost;
  }

  static int GetVectorCost(Instruction *I, unsigned Count,
                           TargetTransformInfo *TTI, TargetLibraryInfo *TLI,
                           LLVMContext &Context) {

    Type *ScalarTy = I->getType();
    if (isa<StoreInst>(I)) {
      ScalarTy = cast<StoreInst>(I)->getValueOperand()->getType();
    } else if (isa<CmpInst>(I)) {
      ScalarTy = cast<CmpInst>(I)->getOperand(0)->getType();
    }

    // if (!isValidElementType(ScalarTy)) return 0;

    VectorType *VecTy = VectorType::get(ScalarTy, Count);

    switch (I->getOpcode()) {
    case Instruction::Store: {
      // We know that we can merge the stores. Calculate the cost.
      unsigned alignment = cast<StoreInst>(I)->getAlignment();
      return TTI->getMemoryOpCost(Instruction::Store, VecTy, alignment, 0, I);
    }
    case Instruction::Load: {
      // Cost of wide load - cost of scalar loads.
      unsigned alignment = cast<LoadInst>(I)->getAlignment();
      return TTI->getMemoryOpCost(Instruction::Load, VecTy, alignment, 0, I);
    }
    case Instruction::PHI: {
      return 0;
    }
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::BitCast: {
      Type *SrcTy = I->getOperand(0)->getType();
      VectorType *SrcVecTy = VectorType::get(SrcTy, Count);
      int Cost = 0;
      if (VecTy != SrcVecTy) {
        Cost = TTI->getCastInstrCost(I->getOpcode(), VecTy, SrcVecTy, I);
      }
      return Cost;
    }
    case Instruction::FCmp:
    case Instruction::ICmp:
    case Instruction::Select: {
      VectorType *MaskTy = VectorType::get(IntegerType::get(Context, 1), Count);
      return TTI->getCmpSelInstrCost(I->getOpcode(), VecTy, MaskTy, I);
    }
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {

      TargetTransformInfo::OperandValueKind Op1VK =
          TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Op2VK =
          TargetTransformInfo::OK_UniformConstantValue;
      TargetTransformInfo::OperandValueProperties Op1VP =
          TargetTransformInfo::OP_None;
      TargetTransformInfo::OperandValueProperties Op2VP =
          TargetTransformInfo::OP_PowerOf2;

      ConstantInt *CInt = dyn_cast<ConstantInt>(I->getOperand(0));
      if (!CInt) {
        Op1VK = TargetTransformInfo::OK_AnyValue;
        Op1VP = TargetTransformInfo::OP_None;
      } else {
        Op1VK = TargetTransformInfo::OK_UniformConstantValue;
        if (CInt->getValue().isPowerOf2())
          Op1VP = TargetTransformInfo::OP_PowerOf2;
        else
          Op1VP = TargetTransformInfo::OP_None;
      }

      CInt = dyn_cast<ConstantInt>(I->getOperand(1));
      if (!CInt) {
        Op2VK = TargetTransformInfo::OK_AnyValue;
        Op2VP = TargetTransformInfo::OP_None;
      } else {
        Op2VK = TargetTransformInfo::OK_UniformConstantValue;
        if (CInt->getValue().isPowerOf2())
          Op2VP = TargetTransformInfo::OP_PowerOf2;
        else
          Op2VP = TargetTransformInfo::OP_None;
      }

      SmallVector<const Value *, 4> Operands(I->operand_values());

      return TTI->getArithmeticInstrCost(I->getOpcode(), VecTy, Op1VK, Op2VK,
                                         Op1VP, Op2VP, Operands);
    }
    case Instruction::GetElementPtr: {
      TargetTransformInfo::OperandValueKind Op1VK =
          TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Op2VK =
          TargetTransformInfo::OK_UniformConstantValue;

      return TTI->getArithmeticInstrCost(Instruction::Add, VecTy, Op1VK, Op2VK);
    }
    case Instruction::Call: {
      CallInst *CI = cast<CallInst>(I);
      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);

      // Calculate the cost of the scalar and std::vector calls.
      SmallVector<Type *, 4> ScalarTys;
      for (unsigned op = 0, opc = CI->getNumArgOperands(); op != opc; ++op)
        ScalarTys.push_back(CI->getArgOperand(op)->getType());

      FastMathFlags FMF;
      if (auto *FPMO = dyn_cast<FPMathOperator>(CI))
        FMF = FPMO->getFastMathFlags();

      SmallVector<Value *, 4> Args(CI->arg_operands());
      return TTI->getIntrinsicInstrCost(ID, CI->getType(), Args, FMF, Count);
      // VecTy->getNumElements());
    }
    default:
      return 0;
    }
  }
};
} // namespace

char MiniSLP::ID = 0;
static RegisterPass<MiniSLP> X("mini-slp", "Mini-SLP Pass");