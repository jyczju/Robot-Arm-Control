import numpy as np

class IKSolver:
    """
    Inverse Kinematics Solver
    :use: IKSolver.solve(target)
    """
    def __init__(self):
        self.q_result = np.zeros((6, 8))
        self.T_Target = np.eye(4)
        self.T_Tool = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.06], [0, 0, 0, 1]])
        self.a2 = 0.185
        self.a3 = 0.17
        self.d1 = 0.23
        self.d2 = -0.054
        self.d3 = 0.0
        self.d4 = 0.077
        self.d5 = 0.077
        self.d6 = 0.0255

    def solve(self, target):
        """
        Inverse Kinematics Solver
        :param target: [x, y, z, rx, ry, rz]
        :return: Joint angle
        """
        self.get_T_target(target)
        self.get_theta() # get theta
        return self.filter()

    def get_T_target(self, target):
        """
        Get the transformation matrix of target
        :param target: [x, y, z, rx, ry, rz]
        """
        x = target[3]
        y = target[4]
        z = target[5]
        rotx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]) # calculate the rotation matrix about x axis
        roty = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]) # calculate the rotation matrix about y axis
        rotz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]]) # calculate the rotation matrix about z axis
        self.T_Target[:3, :3] = np.dot(np.dot(rotx, roty), rotz) # rotation matrix
        self.T_Target[:3, 3] = target[0:3].T # translation matrix
        self.T_Target = np.dot(self.T_Target, np.linalg.inv(self.T_Tool)) # get T_target

    def get_theta(self):
        """
        Calculate theta
        """
        r11, r12, r13, x_p = self.T_Target[0, :]
        r21, r22, r23, y_p = self.T_Target[1, :]
        rho_x = self.d6 * r23 - y_p
        rho_y = self.d6 * r13 - x_p

        # get theta_1 - situation 1
        theta_1 = np.arctan2(rho_x, rho_y) - np.arctan2(-(self.d2 + self.d4), np.sqrt(rho_x**2 + rho_y**2 - (self.d2 + self.d4)**2))
        theta_1 = self.limit_convert(theta_1)
        self.q_result[0, :4] = theta_1

        # get theta_1 - situation 1
        theta_1 = np.arctan2(rho_x, rho_y) - np.arctan2(-(self.d2 + self.d4), -np.sqrt(rho_x**2 + rho_y**2 - (self.d2 + self.d4)**2))
        theta_1 = self.limit_convert(theta_1)
        self.q_result[0, 4:8] = theta_1

        # get theta_5 - situation 1
        theta_5 = np.arcsin(-np.sin(self.q_result[0, 0]) * r13 + np.cos(self.q_result[0, 0]) * r23)
        self.q_result[4, [0, 1]] = theta_5
        if theta_5 < 0:
            self.q_result[4, [2, 3]] = -np.pi - self.q_result[4, 0]
        else:
            self.q_result[4, [2, 3]] = np.pi - self.q_result[4, 0]

        # get theta_5 - situation 2
        theta_5 = np.arcsin(-np.sin(self.q_result[0, 4]) * r13 + np.cos(self.q_result[0, 4]) * r23)
        self.q_result[4, [4, 5]] = theta_5
        if theta_5 < 0:
            self.q_result[4, [6, 7]] = -np.pi - self.q_result[4, 4]
        else:
            self.q_result[4, [6, 7]] = np.pi - self.q_result[4, 4]
    
        self.q_result[5, :] = np.arctan2(-(-np.sin(self.q_result[0, :]) * r12 + np.cos(self.q_result[0, :]) * r22) * np.cos(self.q_result[4, :]), (-np.sin(self.q_result[0, :]) * r11 + np.cos(self.q_result[0, :]) * r21) * np.cos(self.q_result[4, :]))

        for i in range(4): # 4 situations
            T01 = np.array([[np.cos(self.q_result[0, 2*i]), -np.sin(self.q_result[0, 2*i]), 0, 0],
             [np.sin(self.q_result[0, 2*i]), np.cos(self.q_result[0, 2*i]), 0, 0],
             [0, 0, 1, self.d1],
             [0, 0, 0, 1]])
            T45 = np.array([[-np.sin(self.q_result[4, 2*i]), -np.cos(self.q_result[4, 2*i]), 0, 0],
             [0, 0, -1, -self.d5],
             [np.cos(self.q_result[4, 2*i]), -np.sin(self.q_result[4, 2*i]), 0, 0],
             [0, 0, 0, 1]])
            T56 = np.array([[np.cos(self.q_result[5, 2*i]), -np.sin(self.q_result[5, 2*i]), 0, 0],
             [0, 0, -1, -self.d6],
             [np.sin(self.q_result[5, 2*i]), np.cos(self.q_result[5, 2*i]), 0, 0],
             [0, 0, 0, 1]])

            # calculate T14
            T14 = np.dot(np.linalg.inv(T01), self.T_Target)
            T14 = np.dot(T14, np.linalg.inv(T56))
            T14 = np.dot(T14, np.linalg.inv(T45))

            x_p = T14[(0, 3)]
            y_p = T14[(2, 3)]
            # get theta_3
            cos_theta_3 = (x_p ** 2 + y_p ** 2 - self.a2 ** 2 - self.a3 ** 2) / (2 * self.a2 * self.a3)
            if np.abs(cos_theta_3) > 1: # the limit by cos()
                self.q_result[:, 2*i] = 100
                self.q_result[:, 2*i+1] = 100
            else:
                theta_3 = np.arccos(cos_theta_3)                   
                self.q_result[(2, 2*i)] = theta_3
                self.q_result[(2, 2*i+1)] = -theta_3

                # get theta_4
                tmp1 = self.a2 + self.a3 * np.cos(self.q_result[2, [2*i, 2*i+1]])
                tmp2 = self.a3 * np.sin(self.q_result[2, [2*i, 2*i+1]])
                self.q_result[1, [2*i, 2*i+1]] = np.arctan2(tmp1*x_p - tmp2*y_p, tmp2*x_p + tmp1*y_p)

                theta_4 = np.arctan2(-T14[0, 1], T14[0, 0]) - self.q_result[1, [2*i, 2*i+1]] - self.q_result[2, [2*i, 2*i+1]]
                for j in range(2):
                    theta_4[j] = self.limit_convert(theta_4[j])

                self.q_result[3, [2*i, 2*i+1]] = theta_4

    def filter(self):
        """
        Filter the result
        :return: Joint angle, 6 * n np.array (n = 0 - 8)
        """
        q_result = self.q_result
        index = np.where(np.abs(q_result[0, :]) <= 200/180*np.pi)
        q_result = q_result[:, index[0]]
        index = np.where(np.abs(q_result[1, :]) <= 90/180*np.pi)
        q_result = q_result[:, index[0]]
        index = np.where(np.abs(q_result[2, :]) <= 120/180*np.pi)
        q_result = q_result[:, index[0]]
        index = np.where(np.abs(q_result[3, :]) <= 150/180*np.pi)
        q_result = q_result[:, index[0]]
        index = np.where(np.abs(q_result[4, :]) <= 150/180*np.pi)
        q_result = q_result[:, index[0]]
        index = np.where(np.abs(q_result[5, :]) <= np.pi)
        q_result = q_result[:, index[0]]
        return q_result

    def limit_convert(self, theta):
        """
        convert the angle to the range of [-pi, pi]
        :param theta: the angle to be converted
        :return: the converted angle
        """
        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi
        return theta