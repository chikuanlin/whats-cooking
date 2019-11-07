import json

class Cuisine:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', None)
        self.cuisine = kwargs.get('cuisine', None)
        self.ingredients = kwargs.get('ingredients', [])
    
    def __str__(self):
        info = 'Id: {}\nCuisine: {}\nIngredients: '.format(self.id, self.cuisine)
        return info + ', '.join(self.ingredients)

class WhatsCookingDataset:
    def __init__(self, file_path='train.json'):
        self.cuisines = []
        self.id2ingredient = []
        self.ingredient2id = {}
        self.id2cuisine = []
        self.cuisine2id = {}
        with open(file_path, encoding='utf-8', mode = 'r') as json_file:
            for item in json.load(json_file):
                self.cuisines.append(Cuisine(**item))
                if item['cuisine'] not in self.cuisine2id:
                    self.id2cuisine.append(item['cuisine'])
                    self.cuisine2id[item['cuisine']] = len(self.id2cuisine)-1
                for ingredient in item['ingredients']:
                    if ingredient not in self.ingredient2id:
                        self.id2ingredient.append(ingredient)
                        self.ingredient2id[ingredient] = len(self.id2ingredient)-1

    def __len__(self):
        return len(self.cuisines)

    def __getitem__(self, index):
        return self.cuisines[index]

    def save_to_text(self, sorting=True):
        if sorting:
            cuisine_output = self.id2cuisine.copy()
            ingredient_output = self.id2ingredient.copy()
            cuisine_output.sort()
            ingredient_output.sort()
        else:
            cuisine_output = self.id2cuisine
            ingredient_output = self.id2ingredient

        with open('cuisine_labels.txt', 'w', encoding='utf-8') as f:
            for cuisine in cuisine_output:
                f.write(cuisine + '\n')
        with open('ingredients_labels.txt', 'w', encoding='utf-8') as f:
            for ingredient in ingredient_output:
                f.write(ingredient + '\n')

    def load_test_file(self, file_path='test.json'):
        cuisines = []
        with open(file_path, encoding='utf-8', mode = 'r') as json_file:
            for item in json.load(json_file):
                cuisines.append(Cuisine(**item))
        return cuisines


if __name__ == "__main__":
    dataset = WhatsCookingDataset()
    print(dataset[0])
    dataset.save_to_text()
    test = dataset.load_test_file()
    print(test[0])