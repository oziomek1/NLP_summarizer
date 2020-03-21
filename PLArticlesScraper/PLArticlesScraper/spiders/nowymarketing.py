# -*- coding: utf-8 -*-
import scrapy


class NowymarketingSpider(scrapy.Spider):
    name = 'nowymarketing'
    allowed_domains = ['nowymarketing.pl']
    start_urls = ['http://nowymarketing.pl/']

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS = "//article[contains(@class, 'boxArticle')]/a/@href"
    NEXT_PAGE_URL = "//nav[contains(@class, 'paginNav')]/a/@href"

    TITLE_NODE = "//article//h1/text()"
    LEAD_NODE = "//article//div[contains(@class, 'articleLead')]/p/text()"
    TEXT_NODE = "//article//div[contains(@class, 'articleBody')]/h2 | //article//div[contains(@class, 'articleBody')]/p[not(iframe) and not(img)]"

    def parse(self, response):
        # iterate through links to articles
        for quote in response.xpath(self.ARTICLES_URLS):
            yield response.follow(quote, callback=self.parse_article)
            
        #iterate through next page links
        for next_page in response.xpath(self.NEXT_PAGE_URL):
            yield response.follow(next_page, self.parse)
            
    def parse_article(self, response):
        yield {
            'url': response.url,
            'title': response.xpath(self.TITLE_NODE).extract(),
            'lead': response.xpath(self.LEAD_NODE).extract(),
            'text': response.xpath(self.TEXT_NODE).extract(),
        }