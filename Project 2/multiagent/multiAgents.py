# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Base score from gmae
        score = successorGameState.getScore()

        # Food features
        foodList = newFood.asList()
        if foodList:
            # Distance to closest food (using Manhattan distance)
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            closestFoodDist = min(foodDistances)
            # Reciprocal of distance to encourage getting closer to food
            score += 1.0 / (closestFoodDist + 1)

        # Ghost features
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            if ghostPos:  # Ghost might be None if eaten
                distToGhost = manhattanDistance(newPos, ghostPos)
                scaredTime = newScaredTimes[i]
            
                if scaredTime > 0:
                    # Ghost is scared - chase it!
                    if distToGhost > 0:
                        score += 200.0 / (distToGhost + 1)
                else:
                    # Ghost is dangerous - avoid it
                    if distToGhost < 2:
                        score -= 1000  # Very bad - almost certain death
                    elif distToGhost < 4:
                        score -= 100
                    elif distToGhost < 6:
                        score -= 20
    
        # Penalize stopping to encourage movement
        if action == Directions.STOP:
            score -= 10
    
        # Consider remaining food count
        score -= len(foodList) * 10

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # Start the minimax recursion from the root (Pacman's turn)
        def minimax(state, agentIndex, depth):
            # Check terminal states
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            # Pacman's turn (maximizing agent)
            if agentIndex == 0:
                return maxValue(state, agentIndex, depth)
            # Ghosts' turn (minimizing agents)
            else:
                return minValue(state, agentIndex, depth)
        
        def maxValue(state, agentIndex, depth):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            
            v = float('-inf')
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                # Next agent is first ghost (index 1)
                v = max(v, minimax(successor, 1, depth))
            return v
        
        def minValue(state, agentIndex, depth):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            
            v = float('inf')
            nextAgent = agentIndex + 1
            nextDepth = depth
            
            # If we've gone through all ghosts, go back to Pacman and increase depth
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                nextDepth = depth + 1
            
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, minimax(successor, nextAgent, nextDepth))
            return v
        
        # Find the best action for Pacman at the root
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP

        bestAction = legalActions[0]
        bestValue = float('-inf')
        
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, 1, 0)  # Start with first ghost at depth 0
            
            if value > bestValue:
                bestValue = value
                bestAction = action
        
        return bestAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(state, agentIndex, depth, alpha, beta):
            # Check terminal states
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            # Pacman's turn (maximizing agent)
            if agentIndex == 0:
                return maxValue(state, agentIndex, depth, alpha, beta)
            # Ghosts' turn (minimizing agents)
            else:
                return minValue(state, agentIndex, depth, alpha, beta)
        
        def maxValue(state, agentIndex, depth, alpha, beta):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            
            v = float('-inf')
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, alphaBeta(successor, 1, depth, alpha, beta))
                
                if v > beta:
                    return v
                alpha = max(alpha, v)
            
            return v
        
        def minValue(state, agentIndex, depth, alpha, beta):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            
            v = float('inf')
            nextAgent = agentIndex + 1
            nextDepth = depth
            
            # If we've gone through all ghosts, go back to Pacman and increase depth
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                nextDepth = depth + 1
            
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, alphaBeta(successor, nextAgent, nextDepth, alpha, beta))
                
                if v < alpha:
                    return v
                beta = min(beta, v)
            
            return v
        
        # Find the best action for Pacman at the root
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = alphaBeta(successor, 1, 0, alpha, beta)
            
            if value > bestValue:
                bestValue = value
                bestAction = action
                
            if value > beta:
                break
            alpha = max(alpha, value)
        
        return bestAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, agentIndex, depth):
            # Check terminal states
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            # Pacman's turn (maximizing agent)
            if agentIndex == 0:
                return maxValue(state, agentIndex, depth)
            # Ghosts' turn (chance nodes)
            else:
                return expValue(state, agentIndex, depth)
        
        def maxValue(state, agentIndex, depth):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            
            v = float('-inf')
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, expectimax(successor, 1, depth))
            
            return v
        
        def expValue(state, agentIndex, depth):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            
            # Calculate average (uniform distribution)
            nextAgent = agentIndex + 1
            nextDepth = depth
            
            # If we've gone through all ghosts, go back to Pacman and increase depth
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                nextDepth = depth + 1
            
            total = 0
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                total += expectimax(successor, nextAgent, nextDepth)
            
            return total / len(legalActions)
        
        # Find the best action for Pacman at the root
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float('-inf')
        
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 1, 0)
            
            if value > bestValue:
                bestValue = value
                bestAction = action
        
        return bestAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <The eval function combines multiple features to create a composite state evaluation
    game score, distance to food/remaining food, proximity to ghosts, power food, capsules>
    """
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    # Terminal states
    if currentGameState.isWin():
        return score + 10000
    if currentGameState.isLose():
        return score - 10000

    # Food Features
    if foodList:
        closestFoodDist = min(manhattanDistance(pos, f) for f in foodList)
        score += 10.0 / (closestFoodDist + 1)
        score -= 4.0 * len(foodList)

    # Stronger pressure to finish game
    score += 100.0 / (len(foodList) + 1)

    # Ghost Features
    for i, ghostState in enumerate(ghostStates):
        ghostPos = ghostState.getPosition()
        if ghostPos is None:
            continue

        dist = manhattanDistance(pos, ghostPos)
        scaredTime = scaredTimes[i]

        if scaredTime > 0:
            # Chase scared ghosts
            score += 200.0 / (dist + 1)
        else:
            # Strong avoid behavior (no infinities)
            if dist <= 1:
                score -= 2000   # very large penalty
            elif dist <= 2:
                score -= 500
            elif dist <= 4:
                score -= 200

    
    # Capsule Features
    if capsules:
        closestCapsule = min(manhattanDistance(pos, c) for c in capsules)
        if any(t == 0 for t in scaredTimes):
            score += 80.0 / (closestCapsule + 1)

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
