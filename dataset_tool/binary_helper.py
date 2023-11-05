def extract_category_probabilites(probaliteis_list):
    """
    Extract probabilies from categories form list of binary clasifiers and creates list of probalites for all classes
    :param probaliteis_list:
    :return:
    """

def create_all_combinations(array):
  """Creates all possible combinations of an array in tree format.

  Args:
    array: A list of elements.

  Returns:
    A list of lists, where each sublist represents a possible combination of the
    elements in the input array.
  """

  if len(array) == 1:
    return [[array[0]]]

  combinations = []
  for i in range(len(array)):
    sub_combinations = create_all_combinations(array[:i] + array[i + 1:])
    for sub_combination in sub_combinations:
      combinations.append([array[i]] + sub_combination)

  return combinations


def convert_array_to_tree(array):
    """Converts an array to a tree structure.

    Args:
      array: A list of elements.

    Returns:
      A tree structure, represented as a list of lists.
    """

    if len(array) == 2:
        return array

    print(f"Intput array: {array}")
    tree = []
    tree.append(array[0])
    cropped = convert_array_to_tree(array[1:])
    print(f"Cropped array: {cropped}")
    tree.append(cropped)

    return tree


def return_possible_trees(categores):
    """
    Creates all possible trees for given categories
    :param categores: list of categories
    :return: list of trees
    """
    combinations = create_all_combinations(categores)
    trees = []
    for combination in combinations:
        tree = convert_array_to_tree(combination)
        trees.append(tree)
    return trees