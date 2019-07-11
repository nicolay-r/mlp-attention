class AttentionYatianColing2016Config(object):

    __entities_per_context = 2
    __hidden_size = 10

    @property
    def EntitiesPerContext(self):
        return self.__entities_per_context

    @property
    def HiddenSize(self):
        return self.__hidden_size
