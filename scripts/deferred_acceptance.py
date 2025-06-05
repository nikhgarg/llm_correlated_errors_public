import numpy as np
import copy


# Function that takes a list of student preferences, college preferences, and college capacities, and returns the resulting matching
# Students are indexed by [N] = {0, 1, ..., N-1} and colleges are indexed by [C] = {0, 1, ..., C-1}
# Student preference lists have length at most C, with each college in [C] appearing at most once
# The list is in descending order of preference, so the college at index 0 is the most preferred
# College preference lists have length at most N, with each student in [N] appearing at most once
# The list is in descending order of preference, so the student at index 0 is the most preferred
# Parameters:
#   - student_prefs: a length N list of lists
#       student_prefs[i] is the preference list of student i
#   - college_prefs: a length C list of lists
#       college_prefs[c] is the preference list of college c
#   - college_caps: a length C list of ints
#       college_caps[c] is the capacity of college c
# Returns:
#   - student_matches: a length N list of ints
#       student_matches[i] is the college in [C] that student i is matched to, -1 if the student is unmatched
#   - college_matches: a length C list of lists
#       college_matches[c] is a list of length college_caps[c] that contains the students matched to c
#       if college_matches[c][j] = -1, that means the spot is vacant
#       college_matches[c] is sorted according to ascending preference, with any vacant spots at the beginning

def get_match(student_prefs, college_prefs, college_caps):
    # Assert that the number of colleges is consistent in college_prefs and college_caps
    assert len(college_prefs) == len(college_caps), 'college_prefs and college_caps must have the same length'

    num_students = len(student_prefs)    
    num_colleges = len(college_prefs)

    # Assert that preference lists follow the proper format
    for pref_list in student_prefs:
        assert not any(c not in range(num_colleges) for c in pref_list), 'student_prefs contains a preference list with a college not in {0, 1, ..., len(college_prefs)-1}'
        assert len(pref_list) == len(set(pref_list)), 'student_prefs contains a preference list with duplicate colleges'
    for pref_list in college_prefs:
        assert not any(i not in range(num_students) for i in pref_list), 'college_prefs contains a preference list with a student not in {0, 1, ..., len(student_prefs)-1}'
        assert len(pref_list) == len(set(pref_list)), 'college_prefs contains a preference list with duplicate students'

    college_prefs_inv = [[0 for _ in range(num_students)] for _ in range(num_colleges)]
    for c in range(num_colleges):
        for rank, i in enumerate(college_prefs[c]):
            college_prefs_inv[c][i] = rank

    # all students are initially unmatched
    student_matches = [-1 for _ in range(num_students)]

    # all spots in a college are initially vacant, represented by -1
    college_matches = [[-1 for _ in range(college_caps[c])] for c in range(num_colleges)]

    # the list of students that are currently being considered by a college
    college_applicants = [[] for _ in range(num_colleges)]

    # the list of student preference lists containing only colleges the student has yet to apply to
    student_remaining_prefs = copy.deepcopy(student_prefs)

    # continue the matching process while some students remain unmatched and have colleges they have yet to apply to
    while any(student_matches[i] == -1 and len(student_remaining_prefs[i]) > 0 for i in range(num_students)):

        # each unmatched student applies to their top remaining college (unless they have applied to all colleges on their list already)
        for i in range(num_students):
            if student_matches[i] == -1 and len(student_remaining_prefs[i]) > 0:
                # the student applies to their most preferred college left on their list
                college_applicants[student_remaining_prefs[i].pop(0)].append(i)

        # each college considers students who applied to it in the current round and updates its tentative matches
        for c in range(num_colleges):

            # consider each applicant to the college
            while len(college_applicants[c]) > 0:
                i = college_applicants[c].pop(0)

                # update id to represent the index such that college c prefers to student i to all students at or below that index
                id = -1
                while id < college_caps[c] - 1 and (college_matches[c][id+1] == -1 or college_prefs_inv[c][i] < college_prefs_inv[c][college_matches[c][id + 1]]):
                    id += 1
                
                # if student i is better than some of the currently matched students, update the matched students to remain in sorted order
                if id >= 0:
                    # if bottom spot at the college is occupied, the student occupying that spot is unmatched
                    if college_matches[c][0] != -1:
                        student_matches[college_matches[c][0]] = -1
                    
                    # insert student i into the correct sorted position and shift students down accordingly
                    for j in range(id):
                        college_matches[c][j] = college_matches[c][j+1]
                    college_matches[c][id] = i

                    # updated the match for student i
                    student_matches[i] = c
    
    return student_matches, college_matches




# Function that takes values and returns the corresponding preference list, with lower values being more preferred
# Parameters:
#   - values: a 1d list of values or a 2d list of lists of values
# Returns:
#   - prefs: a 1d preference list or a 2d list of preference lists
# Examples:
#   prefs_from_values([0.1, 0.2, 0.3]) = [0, 1, 2]
#   prefs_from_values([[0.1, 0.2, 0.3], [3.1, 1.1, 2.1]) = [[0, 1, 2], [2, 0, 1]]

def prefs_from_values(values):
    prefs = np.argsort(np.array(values)).tolist()
    return prefs




# Function that returns a the list of the rank of the college matched to each student according to the student's preference list
# Parameters:
#   - student_prefs: a length N list of lists
#       student_prefs[i] is the preference list of student i
#   - student_matches: a length N list
#       student_matches[i] is the college that student i is matched to, -1 if the student is not matched
#   - OPTIONAL unmatched_value: int
#       unmatched_value is the rank of a student who is not matched, default 0
# Returns:
#   - ranks: a length N list
#       ranks[i] is the rank of the college a student is matched to according to the student's preference list, unmatched_value if student is not matched
#       therefore, rank[i] = 1 implies student i was matched to their most preferred college

def student_rank_of_match(student_prefs, student_matches, unmatched_value=0):

    ranks = [unmatched_value for _ in range(len(student_prefs))]
    for i in range(len(student_prefs)):
        if student_matches[i] != -1:
            ranks[i] = student_prefs[i].index(student_matches[i]) + 1    
    return ranks




# Function that returns a list of the total value of the students matched to a college according to the college's values for students
# Parameters:
#   - college_values: a length C list of lists of real numbers
#       college_values[c][i] is the value of student i to school c
#   - college_matches: a length C list of lists
#       college_matches[c] is the list of students a college is matched to, college_matches[c][k] = -1 means that spot is vacant
# Returns:
#   - college_match_values: a length C list of ints
#       college_match_values[c] is the total value of the students the college is matched to, according to college_values; vacancies have value 0 for the college.

def college_value_of_match(college_values, college_matches):
    college_match_values = [np.sum([0 if i == -1 else college_values[c][i] for i in college_matches[c]]) for c in range(len(college_values))]
    return college_match_values
