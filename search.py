# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import game
from util import PriorityQueue
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
       
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next_state))
        return next_node
    
    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)
def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    stack=util.Stack()
    stack.push((problem.getStartState(),'',0))
    visited=set()
    finalpath=util.Stack()
    crossRoad=util.Stack()
    if problem.isGoalState(problem.getStartState()):
        return 'stop'
    while not stack.isEmpty():
        curState=stack.pop()
        # print(curState,end=" ")
        visitedChild=0
        visited.add(curState[0])
        if problem.isGoalState(curState[0]):
            finalpath.push(curState)
            break
        finalpath.push(curState)
        noSuccessors=problem.getSuccessors(curState[0])
        for child in noSuccessors:
            if child[0] not in visited :
                stack.push(child)
            else:
                visitedChild+=1
        if (len(noSuccessors)-visitedChild)>=2 :
            crossRoad.push(curState)
            if len(noSuccessors)==4:
                crossRoad.push(curState)
        if len(noSuccessors)==visitedChild:
            temState=finalpath.pop()
            crossState=crossRoad.pop()
            while temState[0]!=crossState[0]:
                temState=finalpath.pop()
            finalpath.push(temState)
    print(finalpath.list)
    finalPath=[x[1] for x in finalpath.list[1:]] 
    return finalPath
    
    # util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue=util.Queue()
    node=Node((problem.getStartState(),'',0))
    queue.push(node)
    visited=set()
    if problem.isGoalState(problem.getStartState()):
        return 'stop'
    finalPath=list()
    while not queue.isEmpty():
        curState=queue.pop()
        if problem.isGoalState(curState.state[0]):
            while curState and curState.state[1]!='':
                finalPath.append(curState.state[1])
                curState=curState.parent
            finalPath.reverse()
            return finalPath
        if curState.state[0] not in visited:
            visited.add(curState.state[0])
            noSuccessors=problem.getSuccessors(curState.state[0])
            # print(noSuccessors,end=" ")
            for child in noSuccessors:
                if child[0] not in visited :
                    queue.push(Node(child,curState))
    
    return None

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    priorityQueue = PriorityQueue()
    visited = set()
    priorityQueue.push(problem.getStartState(),0)
    act = PriorityQueue()
    act.push([], 0)
    while not priorityQueue.isEmpty():
        curState = priorityQueue.pop()
        finalPath = act.pop()
        if problem.isGoalState(curState):
            # print(finalPath)
            return finalPath
        if curState not in visited:
            visited.add(curState) 
            noSuccessors = problem.getSuccessors(curState)
            for child, action, _ in noSuccessors:
                if child not in visited:
                    new_act = finalPath + [action]
                    new_cost = problem.getCostOfActions(new_act)
                    priorityQueue.push(child, new_cost)
                    act.push(new_act, new_cost)
    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = PriorityQueue()
    visited = set()
    priorityQueue.push(problem.getStartState(),0)
    act = PriorityQueue()
    act.push([], 0)
    while not priorityQueue.isEmpty():
        curState = priorityQueue.pop()
        finalPath = act.pop()
        if problem.isGoalState(curState):
            # print(finalPath)
            return finalPath
        if curState not in visited:
            visited.add(curState) 
            noSuccessors = problem.getSuccessors(curState)
            for child, action, _ in noSuccessors:
                if child not in visited:
                    new_act = finalPath + [action]
                    g_cost = problem.getCostOfActions(new_act)
                    h_cost = heuristic(child, problem)
                    f_cost = h_cost+g_cost
                    priorityQueue.push(child, f_cost)
                    act.push(new_act, f_cost)
    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
