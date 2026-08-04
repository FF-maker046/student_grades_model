# 学生成绩预测系统 - 最终版

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体，解决图表中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class StudentGradePredictor:
    """学生成绩预测器 - 最终版"""

    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()  # 特征标准化器
        self.features = ['学习时间(小时/天)', '作业完成率(%)', '课堂出勤率(%)', '前测成绩']
        self.target = '期末成绩'
        self.is_trained = False

    def generate_realistic_data(self, num_samples=400):
        """生成符合现实的模拟学生数据"""
        np.random.seed(0)  # 设置随机种子保证可重复

        print(f"正在生成 {num_samples} 条模拟学生数据...")

        # 1. 学习时间：大部分在3-8小时
        study_hours = np.clip(np.random.normal(4.5, 1.5, num_samples),1, 8)

        # 2. 作业完成率：大部分在75-95%之间
        homework_rate = np.clip(np.random.normal(85, 10, num_samples), 0, 100)

        # 3. 课堂出勤率：大部分在85-100%之间
        attendance_rate = np.clip(np.random.normal(92, 8, num_samples), 0, 100)

        # 4. 前测成绩：0-100之间的正态分布
        pretest_scores = np.clip(np.random.normal(72, 14, num_samples), 0, 100)

        # 创建数据框
        data = {
            '学习时间(小时/天)': study_hours,
            '作业完成率(%)': homework_rate,
            '课堂出勤率(%)': attendance_rate,
            '前测成绩': pretest_scores
        }

        df = pd.DataFrame(data)

        # 计算期末成绩（使用现实的教育关系）
        print("正在计算期末成绩...")

        # 基础权重：前测成绩最重要，作业次之，出勤率再次，学习时间影响最小
        weights = {
            '前测成绩': 0.50,  # 基础能力最重要
            '作业完成率': 0.30,  # 学习态度关键
            '课堂出勤率': 0.15,  # 课堂参与重要
            '学习时间': 0.05  # 时间投入有影响但效率更重要
        }

        # 计算基础成绩
        base_score = (
                weights['前测成绩'] * df['前测成绩'] +
                weights['作业完成率'] * df['作业完成率(%)']  +
                weights['课堂出勤率'] * df['课堂出勤率(%)']  +
                weights['学习时间'] * df['学习时间(小时/天)'] * 12
        )

        # 添加非线性关系和交互效应
        # 1. 学习效率效应：学习时间过长可能效率下降
        efficiency = np.ones(num_samples)
        efficiency[df['学习时间(小时/天)'] > 7] *= 0.98  # 过度学习效率略降

        # 2. 勤奋奖励效应：高作业完成率+高出勤率有额外奖励
        diligence_bonus = ((df['作业完成率(%)'] > 90) & (df['课堂出勤率(%)'] > 95)) * 8

        # 计算最终成绩
        final_score = base_score * efficiency + diligence_bonus

        # 确保成绩在0-100之间
        df['期末成绩'] = np.clip(final_score, 0, 100).round(1)

        # 统计成绩分布
        print("\n成绩分布统计：")
        print(f"  前测成绩: 平均={df['前测成绩'].mean():.1f}, 标准差={df['前测成绩'].std():.1f}")
        print(f"  期末成绩: 平均={df['期末成绩'].mean():.1f}, 标准差={df['期末成绩'].std():.1f}")

        # 计算各分数段人数
        grade_ranges = [(0, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
        grade_labels = ['不及格', '及格', '中等', '良好', '优秀']

        print("\n成绩分布：")
        for (low, high), label in zip(grade_ranges, grade_labels):
            if high == 100:
                count = (df['期末成绩'] >= low).sum()
            else:
                count = ((df['期末成绩'] >= low) & (df['期末成绩'] < high)).sum()
            percentage = count / num_samples * 100
            print(f"  {label}({low}-{high}分): {count}人 ({percentage:.1f}%)")

        return df

    def train_model(self, data):
        """训练预测模型"""
        X = data[self.features]
        y = data[self.target]

        print("\n正在划分训练集和测试集...")
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("正在标准化特征数据...")
        # 特征标准化 - 提高模型性能
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("正在训练线性回归模型...")
        # 训练模型
        self.model.fit(X_train_scaled, y_train)

        # 计算模型性能
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        self.is_trained = True
        return train_r2, test_r2, train_mae, test_mae, X_test_scaled, y_test, y_test_pred

    def predict(self, input_data):
        """预测学生成绩"""
        if not self.is_trained:
            return "错误：请先训练模型！"

        # 确保输入格式正确
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame([input_data], columns=self.features)

        # 标准化输入特征
        input_scaled = self.scaler.transform(input_df)

        prediction = self.model.predict(input_scaled)[0]
        # 确保预测成绩在0-100范围内
        return max(0, min(100, round(prediction, 1)))

    def analyze_features(self):
        """分析各特征对成绩的影响"""
        coefficients = self.model.coef_

        # 创建重要性分析数据框
        importance_df = pd.DataFrame({
            '特征': self.features,
            '标准化系数': [round(coef, 4) for coef in coefficients],
            '对成绩的影响方向': ['正向' if coef > 0 else '负向' for coef in coefficients]
        })

        # 计算相对重要性（百分比）
        abs_coef = np.abs(coefficients)
        importance_df['相对重要性(%)'] = (abs_coef / abs_coef.sum()) * 100

        # 计算实际贡献（考虑特征标准差）
        if hasattr(self.scaler, 'scale_'):
            importance_df['特征标准差'] = self.scaler.scale_
            importance_df['实际贡献值'] = abs_coef * self.scaler.scale_
            total_contrib = importance_df['实际贡献值'].sum()
            importance_df['实际重要性(%)'] = (importance_df['实际贡献值'] / total_contrib) * 100

        return importance_df

    def plot_comprehensive_analysis(self, y_test, y_pred, importance_df):
        """绘制完整的分析图表"""
        fig = plt.figure(figsize=(16, 12))

        # 1. 预测vs实际散点图
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(y_test, y_pred, alpha=0.6, color='steelblue', s=50)
        ax1.plot([0, 100], [0, 100], 'r--', linewidth=2, label='理想预测线')

        # 添加回归线
        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)
        ax1.plot(sorted(y_test), p(sorted(y_test)), "g-", linewidth=2, label=f'实际回归线')

        ax1.set_xlabel('实际期末成绩 (分)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('预测期末成绩 (分)', fontsize=12, fontweight='bold')
        ax1.set_title('预测成绩 vs 实际成绩对比', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.axis([0, 100, 0, 100])

        # 计算并显示R²分数
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.2f}分', transform=ax1.transAxes,
                 fontsize=12, fontweight='bold', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 2. 误差分布直方图
        ax2 = plt.subplot(2, 3, 2)
        errors = y_pred - y_test
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        n, bins, patches = ax2.hist(errors, bins=20, edgecolor='black',
                                    alpha=0.7, color='lightcoral')

        # 添加统计线
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.axvline(x=mean_error, color='blue', linestyle='-', linewidth=2,
                    label=f'平均误差: {mean_error:.2f}分')

        ax2.set_xlabel('预测误差 (预测值 - 实际值)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('学生数量', fontsize=12, fontweight='bold')
        ax2.set_title('预测误差分布', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # 添加误差统计
        error_stats = (f'平均误差: {mean_error:.2f}分\n'
                       f'误差标准差: {std_error:.2f}分\n'
                       f'最大正误差: {max(errors):.2f}分\n'
                       f'最大负误差: {min(errors):.2f}分')

        ax2.text(0.05, 0.95, error_stats, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 3. 特征重要性柱状图
        ax3 = plt.subplot(2, 3, 3)

        if '实际重要性(%)' in importance_df.columns:
            values = importance_df['实际重要性(%)']
        else:
            values = importance_df['相对重要性(%)']

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax3.bar(importance_df['特征'], values,
                       color=colors, edgecolor='black', linewidth=1.5)

        # 在柱状图上添加数值标签
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{value:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax3.set_xlabel('特征变量', fontsize=12, fontweight='bold')
        ax3.set_ylabel('对成绩的影响重要性 (%)', fontsize=12, fontweight='bold')
        ax3.set_title('各特征对期末成绩的影响', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xticklabels(importance_df['特征'], rotation=15, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. 成绩分布对比图
        ax4 = plt.subplot(2, 3, 4)

        # 定义分数段
        bins = [0, 60, 70, 80, 90, 100]
        labels = ['不及格\n(0-59)', '及格\n(60-69)', '中等\n(70-79)', '良好\n(80-89)', '优秀\n(90-100)']

        actual_counts, _ = np.histogram(y_test, bins=bins)
        pred_counts, _ = np.histogram(y_pred, bins=bins)

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax4.bar(x - width / 2, actual_counts, width, label='实际成绩', color='skyblue', alpha=0.8)
        bars2 = ax4.bar(x + width / 2, pred_counts, width, label='预测成绩', color='lightcoral', alpha=0.8)

        ax4.set_xlabel('成绩等级', fontsize=12, fontweight='bold')
        ax4.set_ylabel('学生数量', fontsize=12, fontweight='bold')
        ax4.set_title('成绩分布对比', fontsize=14, fontweight='bold', pad=15)
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # 添加数量标签
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

        # 5. 残差图
        ax5 = plt.subplot(2, 3, 5)

        ax5.scatter(y_pred, errors, alpha=0.6, color='purple', s=50)
        ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax5.axhline(y=mean_error, color='blue', linestyle='-', linewidth=1, label='平均误差线')

        ax5.set_xlabel('预测成绩 (分)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('残差 (预测 - 实际)', fontsize=12, fontweight='bold')
        ax5.set_title('残差分析图', fontsize=14, fontweight='bold', pad=15)
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)

        # 6. 模型系数可视化
        ax6 = plt.subplot(2, 3, 6)

        coefficients = self.model.coef_
        feature_names = self.features

        # 创建系数条形图
        y_pos = np.arange(len(feature_names))
        colors_coef = ['green' if c > 0 else 'red' for c in coefficients]

        bars_coef = ax6.barh(y_pos, coefficients, color=colors_coef, edgecolor='black')

        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(feature_names)
        ax6.set_xlabel('系数值', fontsize=12, fontweight='bold')
        ax6.set_title('模型系数可视化', fontsize=14, fontweight='bold', pad=15)
        ax6.grid(True, alpha=0.3, axis='x')

        # 添加系数值标签
        for i, (bar, coef) in enumerate(zip(bars_coef, coefficients)):
            width = bar.get_width()
            label_x = width + (0.01 if width >= 0 else -0.01)
            ha = 'left' if width >= 0 else 'right'
            ax6.text(label_x, bar.get_y() + bar.get_height() / 2,
                     f'{coef:.3f}', ha=ha, va='center', fontsize=10, fontweight='bold')

        # 调整布局
        plt.suptitle('学生成绩预测系统 - 综合分析报告', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 保存图表
        plt.savefig('学生成绩预测系统_综合分析图.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        print("\n综合分析图表已保存为 '学生成绩预测系统_综合分析图.png'")

        return r2, mae

    def get_detailed_model_info(self):
        """获取详细的模型信息"""
        info = {
            '模型状态': '已训练' if self.is_trained else '未训练',
            '模型类型': '多元线性回归',
            '特征数量': len(self.features),
            '特征名称': self.features,
            '目标变量': self.target,
            '特征标准化': '已应用'
        }

        if self.is_trained:
            info['截距项(常数)'] = f"{self.model.intercept_:.4f}"

            # 系数信息
            coef_dict = {}
            for feature, coef in zip(self.features, self.model.coef_):
                direction = "正向" if coef > 0 else "负向"
                coef_dict[feature] = f"{coef:.4f} ({direction})"
            info['特征系数'] = coef_dict

        return info


def print_section_header(title):
    """打印章节标题"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def validate_input(prompt, min_val, max_val):
    """验证用户输入"""
    while True:
        try:
            value = input(prompt)
            if value.lower() == 'q':
                return None
            value_f = float(value)
            if min_val <= value_f <= max_val:
                return value_f
            else:
                print(f"请输入{min_val}到{max_val}之间的数字")
        except ValueError:
            print("请输入有效的数字")


def main():
    """主程序"""
    print_section_header("学生成绩预测系统 - 最终版")
    print("功能：基于多元线性回归预测学生期末成绩")
    print("成绩范围：0-100分（符合实际情况）")

    # 创建预测器
    predictor = StudentGradePredictor()

    # 1. 生成模拟数据
    print_section_header("1. 数据生成与准备")
    data = predictor.generate_realistic_data(400)
    print(f"\n已成功生成 {len(data)} 条学生记录")

    # 显示前5条数据示例
    print("\n数据示例（前5条）：")
    print(data.head().to_string())

    # 2. 训练模型
    print_section_header("2. 模型训练与评估")
    train_r2, test_r2, train_mae, test_mae, X_test_scaled, y_test, y_test_pred = predictor.train_model(data)

    print(f"\n模型性能评估：")
    print(f"训练集 R² 分数: {train_r2:.4f}")
    print(f"测试集 R² 分数: {test_r2:.4f}")
    print(f"训练集 MAE (平均绝对误差): {train_mae:.2f}分")
    print(f"测试集 MAE (平均绝对误差): {test_mae:.2f}分")

    # 性能评级
    if test_r2 > 0.8:
        rating = "优秀"
    elif test_r2 > 0.7:
        rating = "良好"
    elif test_r2 > 0.6:
        rating = "中等"
    elif test_r2 > 0.5:
        rating = "一般"
    else:
        rating = "较差"

    print(f"\n模型性能评级: {rating}")
    print(f"模型解释力: {test_r2 * 100:.1f}% 的期末成绩变异可以被模型解释")

    # 3. 特征重要性分析
    print_section_header("3. 特征重要性分析")
    importance_df = predictor.analyze_features()
    print("\n特征重要性分析结果：")
    print(importance_df.to_string(index=False))

    print("\n关键发现：")
    for _, row in importance_df.iterrows():
        feature = row['特征']
        if '实际重要性(%)' in row:
            importance = row['实际重要性(%)']
        else:
            importance = row['相对重要性(%)']
        direction = row['对成绩的影响方向']

        if direction == '正向':
            effect = "提高成绩"
        else:
            effect = "降低成绩"

        print(f"{feature}: 影响权重{importance:.1f}%，{effect}")

    # 4. 模型详细信息
    print_section_header("4. 模型详细信息")
    model_info = predictor.get_detailed_model_info()
    for key, value in model_info.items():
        if key == '特征系数':
            print(f"{key}:")
            for feat, coef in value.items():
                print(f"{feat}: {coef}")
        else:
            print(f"{key}: {value}")

    # 5. 示例预测
    print_section_header("5. 示例预测")

    # 定义不同水平的学生案例
    example_cases = [
        {
            'name': '优秀学生',
            'data': {'学习时间(小时/天)': 5.5, '作业完成率(%)': 95, '课堂出勤率(%)': 98, '前测成绩': 88},
            'desc': '学习努力，基础好'
        },
        {
            'name': '中等学生',
            'data': {'学习时间(小时/天)': 4.0, '作业完成率(%)': 80, '课堂出勤率(%)': 90, '前测成绩': 75},
            'desc': '表现一般，有提升空间'
        },
        {
            'name': '后进学生',
            'data': {'学习时间(小时/天)': 2.5, '作业完成率(%)': 65, '课堂出勤率(%)': 75, '前测成绩': 55},
            'desc': '学习时间少，基础薄弱'
        },
        {
            'name': '偏科学生',
            'data': {'学习时间(小时/天)': 6.0, '作业完成率(%)': 90, '课堂出勤率(%)': 85, '前测成绩': 85},
            'desc': '学习努力且基础好'
        }
    ]

    print("\n不同学生类型的预测结果：")
    print("┌────────────┬──────────────────────────────────────┬──────────────┐")
    print("│ 学生类型     │  特征描述                             │ 预测成绩       │")
    print("├────────────┼──────────────────────────────────────┼──────────────┤")

    for case in example_cases:
        prediction = predictor.predict(case['data'])

        # 格式化显示
        name = case['name']
        desc = case['desc']
        pred_str = f"{prediction:.1f}分"

        print(f"│{name:<10}│{desc:<33}│{pred_str:<12}│")

    print("└────────────┴──────────────────────────────────────┴──────────────┘")

    # 6. 交互式预测
    print_section_header("6. 交互式预测")
    print("您可以输入学生信息进行个性化预测（输入'q'退出）")

    while True:
        print("\n" + "-" * 50)
        print("请输入学生信息（所有成绩范围为0-100分）：")

        study_time = validate_input("   学习时间(小时/天, 1-8): ", 1, 8)
        if study_time is None:
            break

        homework = validate_input("   作业完成率(%, 0-100): ", 0, 100)
        if homework is None:
            break

        attendance = validate_input("   课堂出勤率(%, 0-100): ", 0, 100)
        if attendance is None:
            break

        pretest = validate_input("   前测成绩(0-100): ", 0, 100)
        if pretest is None:
            break

        # 创建输入数据
        input_data = {
            '学习时间(小时/天)': study_time,
            '作业完成率(%)': homework,
            '课堂出勤率(%)': attendance,
            '前测成绩': pretest
        }

        # 进行预测
        prediction = predictor.predict(input_data)

        print(f"\n预测结果分析")
        print(f"──────────────────")
        print(f"输入特征：")
        print(f"• 学习时间: {study_time:.1f}小时/天")
        print(f"• 作业完成率: {homework:.1f}%")
        print(f"• 课堂出勤率: {attendance:.1f}%")
        print(f"• 前测成绩: {pretest:.1f}分")
        print(f"\n预测期末成绩: {prediction:.1f}分")

        # 成绩等级判断
        if prediction >= 90:
            grade = "优秀"
            color = "\033[92m"  # 绿色
        elif prediction >= 80:
            grade = "良好"
            color = "\033[94m"  # 蓝色
        elif prediction >= 70:
            grade = "中等"
            color = "\033[93m"  # 黄色
        elif prediction >= 60:
            grade = "及格"
            color = "\033[95m"  # 紫色
        else:
            grade = "不及格"
            color = "\033[91m"  # 红色

        print(f"\n成绩等级: {color}{grade}\033[0m")

        # 学习建议
        print(f"\n个性化学习建议：")
        if prediction < 60:
            print(f"     1. 重点关注前测薄弱知识点（当前{pretest:.1f}分）")
            print(f"     2. 将学习时间增加到至少4小时/天（当前{study_time:.1f}小时）")
            print(f"     3. 确保作业完成率达到80%以上（当前{homework:.1f}%）")
            print(f"     4. 寻求老师或同学帮助，及时解决问题")
        elif prediction < 70:
            print(f"     1. 重点提升前测成绩到70分以上")
            print(f"     2. 保持每天4-5小时有效学习")
            print(f"     3. 提高作业质量和完成率")
            print(f"     4. 制定明确的学习目标和计划")
        elif prediction < 80:
            print(f"     1. 优化学习方法，提高学习效率")
            print(f"     2. 加强中等难度题目的练习")
            print(f"     3. 参与学习小组，互相讨论")
            print(f"     4. 设定85分以上的目标")
        elif prediction < 90:
            print(f"     1. 挑战更高难度的学习内容")
            print(f"     2. 分析错题，避免重复错误")
            print(f"     3. 拓展课外知识，深化理解")
            print(f"     4. 帮助同学，教学相长")
        else:
            print(f"     1. 保持优秀的学习习惯和方法")
            print(f"     2. 参与学科竞赛或研究项目")
            print(f"     3. 深入学习高级课程内容")
            print(f"     4. 分享学习经验，帮助他人进步")

        # 继续预测提示
        print(f"\n输入任意键继续预测，或输入'q'退出")
        if input("选择: ").lower() == 'q':
            break

    # 7. 生成综合分析图表
    print_section_header("7. 可视化分析")
    print("正在生成综合分析图表...")

    try:
        final_r2, final_mae = predictor.plot_comprehensive_analysis(y_test, y_test_pred, importance_df)
        print(f"图表生成完成！")
        print(f"最终模型性能：R² = {final_r2:.4f}, MAE = {final_mae:.2f}分")
    except Exception as e:
        print(f"图表生成失败: {str(e)}")
        print("但程序其他功能正常")

    # 8. 程序结束
    print_section_header("程序运行结束")
    print("生成的文件：")
    print("• 学生成绩预测系统_综合分析图.png - 完整分析图表")
    print("\n 模型总结：")
    print(f"• 模型类型: 多元线性回归")
    print(f"• 数据量: {len(data)} 条学生记录")
    print(f"• 测试集R²分数: {test_r2:.4f}")
    print(f"• 测试集MAE: {test_mae:.2f}分")
    print(f"• 成绩范围: 0-100分（符合实际）")
    print("\n教育应用价值：")
    print("1. 早期识别学习困难学生")
    print("2. 提供个性化学习建议")
    print("3. 帮助教师优化教学方法")
    print("4. 促进学生自我调整学习策略")


if __name__ == "__main__":
    main()