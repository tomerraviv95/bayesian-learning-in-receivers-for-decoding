import numpy as np


class scc:
    def __init__(self, Ein):
        """
        H redundant according to: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4036366
        Cycle count according to: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1564444
        :param Ein:
        """
        self.W, self.U = Ein.shape
        self.g = 1000000
        self.Ng = 0
        self.Ng2 = 0
        self.Ng4 = 0
        self.E = np.copy(Ein)
        self.L_U_0_2 = np.zeros(self.E.shape[0])
        self.L_W_0_2 = np.zeros(self.E.shape[1])
        self.L_U_1_2 = np.zeros(self.E.shape)
        self.P_U_3 = np.zeros(self.E.shape)
        self.L_U_0_4 = np.zeros(self.E.shape[0])

    def count_four_cycles(self, save_Ng=True):
        self.L_U_0_2 = np.matmul(self.E, self.E.T) * np.eye(self.E.shape[0])
        self.L_W_0_2 = np.matmul(self.E.T, self.E) * np.eye(self.E.shape[1])

        P_U_2 = np.matmul(self.E, self.E.T) - self.L_U_0_2

        self.L_U_1_2 = np.matmul(self.E, np.maximum(self.L_W_0_2 - 1, 0))

        self.P_U_3 = np.matmul(P_U_2, self.E) - self.L_U_1_2

        self.L_U_0_4 = np.matmul(self.P_U_3, self.E.T) * np.eye(self.E.shape[0])
        Ng = np.trace(self.L_U_0_4) / 4
        if save_Ng:
            self.Ng = Ng

        return Ng

    def count_six_cycles(self, save_Ng=True):
        P_W_2 = np.matmul(self.E.T, self.E) - self.L_W_0_2

        L_W_1_2 = np.matmul(self.E.T, np.maximum(self.L_U_0_2 - 1, 0))

        P_W_3 = np.matmul(P_W_2, self.E.T) - L_W_1_2

        L_W_0_4 = np.matmul(P_W_3, self.E) * np.eye(self.E.shape[1])
        L_U_2_2 = np.matmul(self.E, L_W_1_2)
        np.fill_diagonal(L_U_2_2, 0)
        L_W_2_2 = np.matmul(self.E.T, self.L_U_1_2)
        np.fill_diagonal(L_W_2_2, 0)

        P_U_4 = np.matmul(self.P_U_3, self.E.T) - self.L_U_0_4 - L_U_2_2

        L_U_3_2 = np.matmul(self.E, L_W_2_2) - self.P_U_3 * self.E - np.matmul(np.maximum(self.L_U_0_2 - 1, 0),
                                                                               self.L_U_1_2)
        L_U_1_4 = np.matmul(self.E, L_W_0_4) - 2 * self.P_U_3 * self.E

        P_U_5 = np.matmul(P_U_4, self.E) - L_U_1_4 - L_U_3_2

        L_U_0_6 = np.matmul(P_U_5, self.E.T) * np.eye(self.E.shape[0])

        Ng = np.trace(L_U_0_6) / 6

        if save_Ng:
            self.Ng = Ng

        return Ng

    def count(self):
        if self.count_four_cycles() > 0:
            self.g = 4
            self.Ng2 = self.count_six_cycles(False)
        elif self.count_six_cycles() > 0:
            self.g = 6
            raise Exception("full support of girth 6 is not fully supported")
        return


def algorithm_2(H_in):
    H_tag = np.copy(H_in)
    r1_star = 0
    r2_star = 0
    scc_t = scc(H_in)
    scc_t.count()
    g_star = scc_t.g
    Ng_star = scc_t.Ng
    Ng2_star = scc_t.Ng2

    while r1_star != -1 or r2_star != -1:
        if r1_star != r2_star:
            H_tag[r2_star] = (H_tag[r2_star] + H_tag[r1_star]) % 2
        r1_star = -1
        r2_star = -1
        for r1 in np.arange(H_in.shape[0]):
            for r2 in np.arange(H_in.shape[0]):
                if r1 != r2:
                    tmp_row = np.copy(H_tag[r2])
                    H_tag[r2] = (H_tag[r2] + H_tag[r1]) % 2
                    scc_t = scc(H_tag)
                    scc_t.count()
                    if scc_t.g > g_star:
                        g_star = scc_t.g
                        r1_star = r1
                        r2_star = r2
                        Ng_star = scc_t.Ng
                        Ng2_star = scc_t.Ng2
                    elif g_star == scc_t.g and Ng_star > scc_t.Ng:
                        r1_star = r1
                        r2_star = r2
                        Ng_star = scc_t.Ng
                        Ng2_star = scc_t.Ng2
                    elif g_star == scc_t.g and Ng_star == scc_t.Ng:
                        if Ng2_star > scc_t.Ng2:
                            r1_star = r1
                            r2_star = r2
                            Ng2_star = scc_t.Ng2
                    H_tag[r2] = tmp_row
    return H_tag
